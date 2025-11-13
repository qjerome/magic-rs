use magic_embed::magic_embed;

#[magic_embed(include=["magic-db/src/magdir"], exclude=["magic-db/src/magdir/der"])]
pub struct CompiledDb;
