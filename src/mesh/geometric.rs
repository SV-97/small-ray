use std::collections::HashMap;

use super::topological::TopologicalHalfEdgeMesh;
use super::{Edge, Vertex};

#[derive(Clone, Debug, Eq, PartialEq)]
struct GeometricHalfEdgeMesh<T> {
    /// All the vertices in the mesh
    vertices: Vec<Vertex<T>>,
    topology: TopologicalHalfEdgeMesh,
}

impl<T: Copy> GeometricHalfEdgeMesh<T> {}
