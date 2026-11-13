//! Address validation for the bounded TP8 diagnostics.
use anyhow::{ensure, Result};
use std::{
    collections::BTreeSet,
    net::{IpAddr, Ipv4Addr, SocketAddr},
};

pub fn management_peers(value: &str) -> Result<[Ipv4Addr; 8]> {
    let peers = value
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let peers: [Ipv4Addr; 8] = peers
        .try_into()
        .map_err(|_| anyhow::anyhow!("provide eight rank-ordered management addresses"))?;
    validate_management_peers(&peers)?;
    Ok(peers)
}

pub(super) fn validate_management_peers(peers: &[Ipv4Addr; 8]) -> Result<()> {
    ensure!(
        peers.iter().all(Ipv4Addr::is_private)
            && peers.iter().copied().collect::<BTreeSet<_>>().len() == 8,
        "provide eight distinct RFC 1918 management addresses"
    );
    Ok(())
}

pub(super) fn validate_http_bind(address: SocketAddr, head: Ipv4Addr) -> Result<()> {
    ensure!(
        head.is_private()
            && address.port() >= 1024
            && matches!(address.ip(), IpAddr::V4(ip) if ip == Ipv4Addr::LOCALHOST || ip == head),
        "bind HTTP to loopback or the configured private rank-zero address"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_peer_count_uniqueness_and_private_scope_without_reordering() {
        let peers = "172.31.240.12,172.31.240.11,172.31.240.13,172.31.240.14,172.31.240.15,172.31.240.16,172.31.240.17,172.31.240.18";
        let parsed = management_peers(peers).unwrap();
        assert_eq!(parsed[0], Ipv4Addr::new(172, 31, 240, 12));
        assert!(management_peers(peers.rsplit_once(',').unwrap().0).is_err());
        assert!(management_peers(&peers.replace("172.31.240.18", "172.31.240.12")).is_err());
        assert!(management_peers(&peers.replace("172.31.240.18", "203.0.113.8")).is_err());
        assert!(management_peers(&peers.replace("172.31.240.18", "127.0.0.1")).is_err());
    }
}
