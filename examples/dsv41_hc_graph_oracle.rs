fn main() -> anyhow::Result<()> {
    anyhow::ensure!(
        std::env::args_os().len() == 1,
        "this device oracle takes no arguments"
    );
    let cases = gb10_inference::dsv41::check_hc_graph_device_replay()?;
    println!(
        "{}",
        serde_json::json!({
            "passed": true,
            "oracle": "device-eager-graph-exact",
            "cases": cases,
            "compared_values": cases.iter().map(|x|x.1).sum::<usize>(),
            "loaded_checkpoint": false,
            "roce_opened": false,
        })
    );
    Ok(())
}
