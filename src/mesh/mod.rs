use std::cmp::Ordering;
use std::fmt::Write as FmtWrite;

pub mod geometric;
pub mod topological;

/// An abstract vertex
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Vertex<T>(pub T);

/// An abstract edge connecting two vertices
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Edge<T> {
    pub start: Vertex<T>,
    pub end: Vertex<T>,
    prev_idx: usize,
    next_idx: usize,
    twin_idx: usize,
}

impl TryFrom<UnfinishedEdge> for Edge<usize> {
    type Error = ();
    fn try_from(pre_edge: UnfinishedEdge) -> Result<Self, Self::Error> {
        Ok(Edge {
            start: Vertex(pre_edge.start_idx),
            end: Vertex(pre_edge.end_idx),
            prev_idx: pre_edge.prev_idx.ok_or(())?,
            next_idx: pre_edge.next_idx.ok_or(())?,
            twin_idx: pre_edge.twin_idx.ok_or(())?,
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct UnfinishedEdge {
    start_idx: usize,
    end_idx: usize,
    prev_idx: Option<usize>,
    next_idx: Option<usize>,
    twin_idx: Option<usize>,
}

impl UnfinishedEdge {
    /// Converts collection of edges to graphviz dot format
    pub fn to_dot(edges: &[Self]) -> String {
        let mut ret: String = "digraph G {{".into();
        for edge in edges.iter() {
            writeln!(&mut ret, "    {} -> {};", edge.start_idx, edge.end_idx).unwrap();
        }
        writeln!(&mut ret, "}}").unwrap();
        ret
    }
}

impl PartialOrd for UnfinishedEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.end_idx == other.start_idx {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}
