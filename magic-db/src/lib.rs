use magic_embed::magic_embed;

#[magic_embed(include=["magic-db/src/magdir"], exclude=["magic-db/src/magdir/der"])]
pub struct CompiledDb;

#[cfg(test)]
mod test {
    use crate::CompiledDb;
    use std::{env, fs::File};

    #[test]
    fn test_compiled_db() {
        let db = CompiledDb::open().unwrap();
        let mut exe = File::open(env::current_exe().unwrap()).unwrap();
        let magic = db.magic_first(&mut exe, None).unwrap();
        println!("{}", magic.message());
        assert!(!magic.is_default())
    }
}
