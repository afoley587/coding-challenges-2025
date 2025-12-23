package main

import (
	"encoding/json"
	"fmt"
	"net"
	"os"

	"github.com/containernetworking/cni/pkg/skel"
	"github.com/containernetworking/cni/pkg/types"
	current "github.com/containernetworking/cni/pkg/types/100"
	"github.com/containernetworking/cni/pkg/version"
	"github.com/containernetworking/plugins/pkg/ip"
	"github.com/containernetworking/plugins/pkg/ipam"
	"github.com/containernetworking/plugins/pkg/ns"
	"github.com/vishvananda/netlink"
)

// NetConf describes the custom bridge configuration supplied via stdin.
type NetConf struct {
	types.NetConf
	MTU               int     `json:"mtu"`
	Masquerade        bool    `json:"masquerade"`
	MasqueradeBackend *string `json:"masqueradeBackend,omitempty"`
}

func cmdAdd(args *skel.CmdArgs) error {
	conf := &NetConf{}

	if err := json.Unmarshal(args.StdinData, conf); err != nil {
		return fmt.Errorf("parse netconf: %w", err)
	}

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

	hostNS, err := ns.GetCurrentNS()
	if err != nil {
		return fmt.Errorf("get host ns: %w", err)
	}
	defer hostNS.Close()

	hostInterface := &current.Interface{}
	containerInterface := &current.Interface{}

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

	return types.PrintResult(currentResult, conf.CNIVersion)
}

func cmdCheck(args *skel.CmdArgs) error {
	conf := &NetConf{}
	if err := json.Unmarshal(args.StdinData, conf); err != nil {
		return fmt.Errorf("parse netconf: %w", err)
	}

	if conf.IPAM.Type != "" {
		if err := ipam.ExecCheck(conf.IPAM.Type, args.StdinData); err != nil {
			return fmt.Errorf("ipam check failed: %w", err)
		}
	}

	return nil
}

func cmdDel(args *skel.CmdArgs) error {
	conf := &NetConf{}
	if err := json.Unmarshal(args.StdinData, conf); err != nil {
		return fmt.Errorf("parse netconf: %w", err)
	}

	if conf.IPAM.Type != "" {
		if err := ipam.ExecDel(conf.IPAM.Type, args.StdinData); err != nil {
			return fmt.Errorf("ipam cleanup failed: %w", err)
		}
	}

	if args.Netns != "" {
		var ipnets []*net.IPNet

		err := ns.WithNetNSPath(args.Netns, func(netns ns.NetNS) error {
			// Inspect link existence (roughly: `ip link show dev <args.IfName>`)
			_, err := netlink.LinkByName(args.IfName)

			if err != nil {
				return fmt.Errorf("couldn't fetch link (%s): %w", args.IfName, err)
			}

			// Delete addresses from the link and remove the link.
			// This helper typically does the equivalent of:
			//   ip addr flush dev <args.IfName>
			//   ip link del <args.IfName>
			//
			// (Exact order/behavior depends on the helper implementation.)
			ipnets, err = ip.DelLinkByNameAddr(args.IfName)

			if err != nil && err != ip.ErrLinkNotFound {
				return fmt.Errorf("link cleanup failed: %w", err)
			}

			return nil
		})

		if err != nil {
			return fmt.Errorf("namespace cleanup failed: %w", err)
		}

		if conf.Masquerade {
			// Remove NAT rules (iptables/nftables), roughly equivalent to deleting the
			// POSTROUTING MASQUERADE rules that were created in ADD.
			if err := ip.TeardownIPMasqForNetworks(ipnets, conf.Name, args.IfName, args.ContainerID); err != nil {
				return fmt.Errorf("masq cleanup failed: %w", err)
			}
		}
	}

	return nil
}

func main() {
	skel.PluginMain(cmdAdd, cmdCheck, cmdDel, version.All, "Custom bridge CNI")
}
