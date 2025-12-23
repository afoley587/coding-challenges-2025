# Container Network Interface Plugins Aren't Scary

## Building and Installing a Custom CNI Into A Local K8s Cluster

Kubernetes networking handles a lot of this for us:

- Assigning Pod IP addresses
- Sending packets across nodes
- Handling ingress/egress
- etc.

Kubernetes doesn't do any of this itself.
Instead, Kubernetes uses a standard interface called the
Container Network Interface (CNI).
The CNI handles the tasks we described above and kubernetes can simply
use the interface.

In this post, we're going to expand upon that by:

1. Explaining what a Container Network Interface (CNI) is

1. Reviewing container networking fundamentals

1. Breaking down what a CNI plugin actually does

1. Walking through a custom CNI configuration

1. Building a custom bridge-based CNI in Go

1. Inspecting the veth pair on both the host and inside the Pod

Hopefully you'll get a better understanding of kubernetes networking at the
end of this post!

## 1. What Is a Container Network Interface (CNI)?

CNI is a
[specification](https://www.cni.dev/docs/spec/),
not a daemon or service.
What we will build is actually a CNI plugin, or a binary
that implements the spec.

The spec itself defines:

* How a container runtime asks for networking
* How a plugin responds
* What JSON data is exchanged
* What lifecycle commands exist

In this project, we will use the
[cni/pkg/skel](https://pkg.go.dev/github.com/containernetworking/cni/pkg/skel)
package to help us adhere to this spec.


You might have heard of some popular CNIs such as Calico, Cilium, and Flannel.
These are just implementations of the CNI spec.

## 2. Container Networking Basics

Containers are just Linux processes running in isolated Linux
(not kubernetes) namespaces.

Each Pod gets a few things:

* A network namespace
* A routing table (in the above namespace)
* A interfaces (in the above namespace)

With this being said, the pod runs on a host node.
The host itself will need a few things so that:

- ingress gets routed to the pod's interfaces
- egress can reach the internet
- etc.

Container technologies usually use a
[veth pair](https://man7.org/linux/man-pages/man4/veth.4.html)
where one side of the veth lives in the container namespace and one end
lives in the host namespace.

There might be additional networking steps that need to be taken.
For example, we might need to
[masquerade](https://www.gsp.com/cgi-bin/man.cgi?section=3&topic=libalias)
IP addresses.

The CNI plugin will handle all of the above.
By default, the network namespace will be empty and the CNI plugin will
create the Veth pair, configure masquerading/NAT, and even more.

## 3. What Does a CNI Plugin Do?

A CNI plugin implements the commands in the
[CNI spec](https://www.cni.dev/docs/spec/#cni-operations)
and is invoked by kubelet.
We will focus on the three commands below:

| Command | Purpose              |
| ------- | -------------------- |
| `ADD`   | Create networking    |
| `CHECK` | Validate networking  |
| `DEL`   | Tear down networking |

Each invocation receives:

* Pod network namespace path
* Interface name (usually `eth0`)
* JSON configuration via stdin

The CNI will use this information to deploy the relevant network
infrastructure on the host and container namespaces.

When kubernetes sees a new pod, it will invoke the CNI plugin and
run the `ADD` command of the plugin.
Kubernetes will pass a lot of data to the CNI.
For a full list of information passed to the CNI,
see the
[go docs](https://pkg.go.dev/github.com/containernetworking/cni/pkg/skel#CmdArgs).
One of the arguments is a JSON blob to the CNI via `stdin` which might
look something like the below.
This JSON comes from our configuration file, a `.conflist`.
We will discuss these files in the next section.

```json
{
  "cniVersion": "1.0.0",
  "name": "custom-bridge",
  "type": "custom-bridge",
  "bridge": "cnibr0",
  "mtu": 1450,
  "masquerade": true,
  "masqueradeBackend": "iptables",
  "ipam": {
    "type": "host-local",
    "ranges": [
      [
        { "subnet": "10.88.0.0/24" }
      ]
    ],
    "routes": [
      { "dst": "0.0.0.0/0" }
    ]
  }
}
```

When kubernetes sees a pod deleting/removed it will invoke the
CNI plugin and run the `DEL` command to cleanup any unused resources.

## 4. Understanding the CNI Configuration

Configuration files for networks are saved in
`/etc/cni/net.d/` on the host machines.
When the namespace is created, the container runtime will
read that directory in lexicographical order.
Our example file, `00-customcni.conflist` would be loaded first
because of the `00-` prefix.

```json
{
  "cniVersion": "1.0.0",
  "name": "custom-bridge",
  "plugins": [
    {
      "type": "custom-bridge",
      "bridge": "cnibr0",
      "mtu": 1450,
      "masquerade": true,
      "masqueradeBackend": "iptables",
      "ipam": {
        "type": "host-local",
        "routes": [
          { "dst": "0.0.0.0/0" }
        ],
        "ranges": [
          [{ "subnet": "10.88.0.0/24" }]
        ]
      }
    }
  ]
}
```

The key fields of this configuration are:

* `type` – The name of the CNI plugin binary in `/opt/cni/bin`
* `bridge` – Linux bridge created on the host
* `mtu` – Maximum Transmission Unit applied to both ends of the veth
* `ipam` – A separate plugin that allocates IPs and routes that
           our plugin will use

A quick note is that we will use the
[IPAM](https://pkg.go.dev/github.com/lstoll/cni/pkg/ipam)
plugin to:

* Allocate IP addresses
* Return gateways and routes
* Track and release IP leases

## 5. Writing the CNI Plugin

As noted above, our CNI plugin is going to implement three
commands:

- `ADD`
- `CHECK`
- `DEL`

For brevity, we will only discuss the `ADD` command in
depth.
The full code can be found in my
[GitHub repo](https://github.com/afoley587/coding-challenges-2025/tree/main/kubernetes/custom-cni)!

### `ADD`

At a high level, our `ADD` command will:

1. Parse the CNI configuration and arguments

1. Define the new IPAM interface

1. Create the new network interface within the container namespace
   and assign IP addresses/routes

1. Create the new network interface within the host namespace
   and assign IP addresses/routes

1. Configure a NAT/Masquerade setup on the host

Let's take a look at the code.

#### Parse the CNI configuration and arguments

The first thing we need to do is to read the configuration
from STDIN:

```go
type NetConf struct {
	types.NetConf
	Bridge            string  `json:"bridge"`
	MTU               int     `json:"mtu"`
	Masquerade        bool    `json:"masquerade"`
	MasqueradeBackend *string `json:"masqueradeBackend,omitempty"`
}

func cmdAdd(args *skel.CmdArgs) error {
	conf := &NetConf{}

	if err := json.Unmarshal(args.StdinData, conf); err != nil {
		return fmt.Errorf("parse netconf: %w", err)
	}
   // The rest of the function
}
```

We can now use the network configuration via dot syntax
and have verified that the data we got is valid.

### Define the new IPAM interface

Next, we need to define the IPAM interface:

```go
func cmdAdd(args *skel.CmdArgs) error {

   // The rest of the function
	result, err := ipam.ExecAdd(conf.IPAM.Type, args.StdinData)
	if err != nil {
		return fmt.Errorf("ipam allocation failed: %w", err)
	}
	defer func() {
		if err != nil {
			_ = ipam.ExecDel(conf.IPAM.Type, args.StdinData)
		}
	}()

	currentResult, err := current.NewResultFromResult(result)
	if err != nil {
		return fmt.Errorf("convert result: %w", err)
	}
   // The rest of the function
}
```

This `currentResult` is a
[result type](https://pkg.go.dev/github.com/containernetworking/cni/pkg/types/100#Result)
and might look something like:

```json
{
  "ips": [
    {
      "address": "10.88.0.12/24",
      "gateway": "10.88.0.1"
    }
  ],
  "routes": [
    { "dst": "0.0.0.0/0" }
  ]
}
```

However, up to here, nothing has been applied yet.

### Create the new network interface within the container namespace

Next, we need to apply the IPAM result within the container
network namespace and assign IP addresses and routes to the
interfaces.
I will try to comment the `ip` Linux equivalents because that
was very helpful for me:

```go
func cmdAdd(args *skel.CmdArgs) error {
   // The rest of the function

   // Get the host namespace and save for later
	hostNS, err := ns.GetCurrentNS()
	if err != nil {
		return fmt.Errorf("get host ns: %w", err)
	}
	defer hostNS.Close()

	hostInterface := &current.Interface{}
	containerInterface := &current.Interface{}

	// The commands within this closure will be executed
   // in the container network namespace
	err = ns.WithNetNSPath(args.Netns, func(netns ns.NetNS) error {

		// Set up a veth pair:
		// - One end stays in the container netns as <args.IfName>
		// - The peer is moved into the host netns
		//
		// Equivalent to roughly:
		//   ip link add <args.IfName> type veth peer name <hostVethName>
		//   ip link set <hostVethName> netns <hostNS>
		//   ip link set dev <args.IfName> mtu <conf.MTU>
		//   ip link set dev <hostVethName> mtu <conf.MTU>
		hostVeth, contVeth, err := ip.SetupVeth(args.IfName, conf.MTU, "", hostNS)
		if err != nil {
			return fmt.Errorf("setup veth: %w", err)
		}
		hostInterface.Name = hostVeth.Name
		hostInterface.Mac = hostVeth.HardwareAddr.String()
		containerInterface.Name = contVeth.Name
		containerInterface.Mac = contVeth.HardwareAddr.String()

		for _, ipc := range currentResult.IPs {
			// All addresses apply to the container veth interface
			ipc.Interface = current.Int(1)
		}

		currentResult.Interfaces = []*current.Interface{hostInterface, containerInterface}

		// Apply IP configuration to the container interface:
		// - add IP address(es)
		// - add routes from the CNI result
		// - set default route via gateway (if provided)
		//
		// Equivalent to combinations of:
		//   ip addr add <IP/CIDR> dev <args.IfName>
		//   ip route add <dst> via <gw> dev <args.IfName>    (and/or)
		//   ip route add default via <gw> dev <args.IfName>
		if err := ipam.ConfigureIface(args.IfName, currentResult); err != nil {
			return fmt.Errorf("configure iface (%s): %w", args.IfName, err)
		}

		// Lookup the container veth by name (no direct `ip` equivalent; inspection-wise):
		//   ip link show dev <args.IfName>
		containerVeth, err := netlink.LinkByName(args.IfName)
		if err != nil {
			return fmt.Errorf("lookup container veth: %w", err)
		}

		// Bring container veth UP:
		//   ip link set dev <args.IfName> up
		if err := netlink.LinkSetUp(containerVeth); err != nil {
			return fmt.Errorf("bring container veth up: %w", err)
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("configuring container interfaces: %w", err)
	}
   // The rest of the function
}
```

At this point, one side of the Veth pair has been configured.
But, if we were to inspect things at this point, we would see
that the link doesn't have any access to network anywhere.

### Create the new network interface within the host namespace

We now need to configure the other half of the Veth pair.
This should look very similar to the container veth configuration.

```go
func cmdAdd(args *skel.CmdArgs) error {
   // The rest of the function

	// Lookup host veth (inspection-wise: `ip link show dev <hostVethName>`)
	hostVeth, err := netlink.LinkByName(hostInterface.Name)
	if err != nil {
		return fmt.Errorf("lookup host veth: %w", err)
	}

	for _, ipc := range currentResult.IPs {
      maskLen := 128
      if ipc.Address.IP.To4() != nil {
         maskLen = 32
      }

		// Add an address to the host veth: you’re adding the *gateway IP* (/32 or /128)
		// on the host-side veth. This is a common point-to-point-ish setup.
		//
		// Equivalent to:
		//   ip addr add <ipc.Gateway>/<32|128> dev <hostVethName>
		ipn := &net.IPNet{
			IP:   ipc.Gateway,
			Mask: net.CIDRMask(maskLen, maskLen),
		}
		addr := &netlink.Addr{IPNet: ipn, Label: ""}
		if err = netlink.AddrAdd(hostVeth, addr); err != nil {
			return fmt.Errorf("failed to add IP addr (%#v) to veth: %v", ipn, err)
		}

		// Add a host route to the container IP via the host veth.
		// With a veth, this typically becomes a direct route like:
		//
		// Equivalent to:
		//   ip route add <containerIP>/<32|128> dev <hostVethName>
		//
		// (No "via" gateway because it’s directly reachable over that link.)
		ipn = &net.IPNet{
			IP:   ipc.Address.IP,
			Mask: net.CIDRMask(maskLen, maskLen),
		}

        if err = ip.AddHostRoute(ipn, nil, hostVeth); err != nil && !os.IsExist(err) {
			return fmt.Errorf("failed to add route on host: %v", err)
		}
	}

	// Bring host veth UP:
	//   ip link set dev <hostVethName> up
	if err := netlink.LinkSetUp(hostVeth); err != nil {
		return fmt.Errorf("bring host veth up: %w", err)
	}

   // The rest of the function
}
```

At this point, our Veth pair has been configured.
However, we will have an issue reaching external sites.
If we started a TCP dump on the host and then ran a Ping from
the container, we might see something like

```shell
docker@minikube:~$ sudo tcpdump -i any -n icmp
tcpdump: data link type LINUX_SLL2
tcpdump: verbose output suppressed, use -v[v]... for full protocol decode
listening on any, link-type LINUX_SLL2 (Linux cooked v2), snapshot length 262144 bytes
11:58:25.044603 vethb18d2fd8 In  IP 10.88.0.3 > 8.8.8.8: ICMP echo request, id 21, seq 0, length 64
11:58:25.044683 eth0  Out IP 10.88.0.3 > 8.8.8.8: ICMP echo request, id 21, seq 0, length 64
11:58:26.049556 vethb18d2fd8 In  IP 10.88.0.3 > 8.8.8.8: ICMP echo request, id 21, seq 1, length 64
11:58:26.049595 eth0  Out IP 10.88.0.3 > 8.8.8.8: ICMP echo request, id 21, seq 1, length 64
```

The issue is that our container traffic is being forwarded to the external network with the
container's IP address (`10.88.0.3`) as the source, but there's no return path
because `10.88.0.0/24` is not routable on the internet.
To fix that, need to set up SNAT (Source NAT) so that outbound traffic
from our containers appears to come from the host's IP address.

### Configure a NAT/Masquerade setup on the host

Let's fix the issue and allow external connectivity by adding
IP Masquerading:

```go
func cmdAdd(args *skel.CmdArgs) error {
   // The rest of the function

	if conf.Masquerade {
		// Setup IP masquerading (NAT) for the container IPs.
		//
		// This is NOT an `ip` command; it corresponds to iptables (legacy) or nftables rules.
		// Roughly equivalent to something like:
		//   iptables -t nat -A POSTROUTING -s <podCIDR_or_IP> ! -d <podCIDR_or_clusterCIDR> -j MASQUERADE
		//
		// Exact rules/chain names vary by backend and CNI library implementation.
		ipns := []*net.IPNet{}
		for _, ip := range currentResult.IPs {
			ipns = append(ipns, &ip.Address)
		}

		if err = ip.SetupIPMasqForNetworks(conf.MasqueradeBackend, ipns, conf.Name, args.IfName, args.ContainerID); err != nil {
			return fmt.Errorf("set up masq: %w", err)
		}
	}
   // The rest of the function
}
```

The above will add IPTable rules for IP Masquerading.
After the pod is created, we could check the IPTable rules with
the following command, which we will do in the next section.

```shell
minikube ssh sudo iptables -t nat -L -n --line-numbers | grep 10.88
```

## 6. Inspecting the Veth Pair (Host and Pod)

Let's now run our CNI plugin.
I am going to use
[skaffold](https://skaffold.dev/)
to automate the build and deployment of the CNI to a local
kubernetes (minikube) cluster.

If you're using the
[GitHub repo](https://github.com/afoley587/coding-challenges-2025/tree/main/kubernetes/custom-cni),
you can use the `Makefile`.

In one terminal, let's run `skaffold` and `minikube`:

```shell
# Start the minikube cluster
make minikube-start

# build and deploy the CNI with skaffold
make skaffold
```

We can verify the CNI was deployed properly by running a few commands
in another terminal:

```shell
# Verify the pod was deployed
% kubectl get pod -n kube-system | grep custom
custom-bridge-cni-l42wl            1/1     Running   0             23s

# Verify the CNI configuration is on the node
% minikube ssh -- ls -l /etc/cni/net.d
total 20
-rw-r--r-- 1 root root 392 Dec 23 16:02 00-customcni.conflist <----------------
-rw-r--r-- 1 root root 496 Dec 23 15:42 1-k8s.conflist
-rw-r--r-- 1 root root 438 Jun 14  2023 100-crio-bridge.conf.mk_disabled
-rw-r--r-- 1 root root  78 Dec 23 16:01 200-loopback.conf
-rw-r--r-- 1 root root 639 Dec 26  2021 87-podman-bridge.conflist.mk_disabled

# Verify our CNI binary is on the node
% minikube ssh -- ls -l /opt/cni/bin
total 83016
-rwxr-xr-x 1 root root  4033874 Dec  4  2023 bandwidth
-rwxr-xr-x 1 root root  4593883 Dec  4  2023 bridge
-rwxr-xr-x 1 root root  5307376 Dec 23 16:02 custom-bridge <----------------
-rwxr-xr-x 1 root root 10648777 Dec  4  2023 dhcp
-rwxr-xr-x 1 root root  4229957 Dec  4  2023 dummy
-rwxr-xr-x 1 root root  4623487 Dec  4  2023 firewall
-rwxr-xr-x 1 root root  4138302 Dec  4  2023 host-device
-rwxr-xr-x 1 root root  3484108 Dec  4  2023 host-local
-rwxr-xr-x 1 root root  4235495 Dec  4  2023 ipvlan
-rwxr-xr-x 1 root root  3642002 Dec  4  2023 loopback
-rwxr-xr-x 1 root root  4244980 Dec  4  2023 macvlan
-rwxr-xr-x 1 root root  4030506 Dec  4  2023 portmap
-rwxr-xr-x 1 root root  4408481 Dec  4  2023 ptp
-rwxr-xr-x 1 root root  3831268 Dec  4  2023 sbr
-rwxr-xr-x 1 root root  3150686 Dec  4  2023 static
-rwxr-xr-x 1 root root  4387761 Dec  4  2023 tap
-rwxr-xr-x 1 root root  3732696 Dec  4  2023 tuning
-rwxr-xr-x 1 root root  4232958 Dec  4  2023 vlan
-rwxr-xr-x 1 root root  4006316 Dec  4  2023 vrf
```

Let's now launch a pod and check it's networking:

```shell
% kubectl run cni-smoke --image=busybox:1.37 --restart=Never --command -- sleep 3600
pod/cni-smoke created
% kubectl exec -it cni-smoke -- sh
/ # ip a
.
.
.
11: eth0@if22: <BROADCAST,MULTICAST,UP,LOWER_UP,M-DOWN> mtu 1450 qdisc noqueue qlen 1000
    link/ether 06:f8:91:f0:a2:73 brd ff:ff:ff:ff:ff:ff
    inet 10.88.0.2/24 brd 10.88.0.255 scope global eth0
       valid_lft forever preferred_lft forever
    inet6 fe80::4f8:91ff:fef0:a273/64 scope link
       valid_lft forever preferred_lft forever
/ # ip route
default via 10.88.0.1 dev eth0
10.88.0.0/24 dev eth0 scope link  src 10.88.0.2
/ # ping -c1 10.88.0.1
PING 10.88.0.1 (10.88.0.1): 56 data bytes
64 bytes from 10.88.0.1: seq=0 ttl=64 time=1.528 ms

--- 10.88.0.1 ping statistics ---
1 packets transmitted, 1 packets received, 0% packet loss
round-trip min/avg/max = 1.528/1.528/1.528 ms
/ # ping -c1 8.8.8.8
PING 8.8.8.8 (8.8.8.8): 56 data bytes
64 bytes from 8.8.8.8: seq=0 ttl=62 time=21.446 ms

--- 8.8.8.8 ping statistics ---
1 packets transmitted, 1 packets received, 0% packet loss
round-trip min/avg/max = 21.446/21.446/21.446 ms
```

We can see the matching Veth on the host as well:

```shell
% minikube ssh
docker@minikube:~$ ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host
       valid_lft forever preferred_lft forever
.
.
.
22: vethc1130b3d@if11: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1450 qdisc noqueue state UP group default
    link/ether 56:c3:56:41:5b:9a brd ff:ff:ff:ff:ff:ff link-netnsid 2
    inet 10.88.0.1/32 scope global vethc1130b3d
       valid_lft forever preferred_lft forever
    inet6 fe80::54c3:56ff:fe41:5b9a/64 scope link
       valid_lft forever preferred_lft forever
docker@minikube:~$ ping -c1 10.88.0.2
PING 10.88.0.2 (10.88.0.2) 56(84) bytes of data.
64 bytes from 10.88.0.2: icmp_seq=1 ttl=64 time=0.383 ms

--- 10.88.0.2 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 0.383/0.383/0.383/0.000 ms
docker@minikube:~$ sudo iptables -t nat -L -n --line-numbers | grep 10.88
6    CNI-178398eb400037331198c6f7  all  --  10.88.0.2            0.0.0.0/0            /* name: "custom-bridge" id: "9fda512a44eb8d8fb6661ff925602c571494e583170e2b2bf666655159e975b4" */
1    ACCEPT     all  --  0.0.0.0/0            10.88.0.0/24         /* name: "custom-bridge" id: "9fda512a44eb8d8fb6661ff925602c571494e583170e2b2bf666655159e975b4" */
```

We can see, from both ends, that the Veth pair is running
as expected and is forwarding traffic to and from the pod.

Thanks for following along with me!
If you liked the post, please comment or clap!
All code is public and available on my
[GitHub repo](https://github.com/afoley587/coding-challenges-2025/tree/main/kubernetes/custom-cni)!
