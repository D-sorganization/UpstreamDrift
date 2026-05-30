#[test]
fn debug_db_error() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::write(
        root.join("hello.py"),
        b"class Greeter:\n    def say_hi(self):\n        pass\n",
    )
    .unwrap();

    let res = upstream_codemap::indexer::rebuild(root, None);
    println!("REBUILD RESULT: {:?}", res);
    if let Err(e) = res {
        panic!("REBUILD FAILED WITH ERROR: {:?}", e);
    }
}
