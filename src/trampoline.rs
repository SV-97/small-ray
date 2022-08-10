pub trait Trampoline<T, S> {
    fn run_til_end(self) -> S;
}

/// Enum to support efficient implementation of tailrecursive functions
pub enum FuncTrampoline<T, S> {
    Val(S),
    Thunk(fn(T) -> Self, T),
}

/// Enum to support efficient implementation of tailrecursive closures
/// If possible you should prefer `FuncTrampoline` as it'll be more efficient
pub enum ClosureTrampoline<T, S> {
    Val(S),
    Thunk(Box<dyn Fn(T) -> Self>, T),
}

impl<T, S> Trampoline<T, S> for FuncTrampoline<T, S> {
    fn run_til_end(self) -> S {
        let mut tramp = self;
        loop {
            match tramp {
                Self::Val(s) => return s,
                Self::Thunk(f, arg) => tramp = f(arg),
            }
        }
    }
}

impl<T, S> Trampoline<T, S> for ClosureTrampoline<T, S> {
    fn run_til_end(self) -> S {
        let mut tramp = self;
        loop {
            match tramp {
                Self::Val(s) => return s,
                Self::Thunk(f, arg) => tramp = f(arg),
            }
        }
    }
}
