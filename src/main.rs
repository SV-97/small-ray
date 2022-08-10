#![allow(dead_code)]
#![recursion_limit = "1025"]
use std::{cmp::Ordering, collections::HashMap, mem::MaybeUninit, ptr::addr_of_mut};

use itertools::Itertools;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Vertex<T>(T);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Edge<T> {
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
    pub fn to_dot(edges: &[Self]) {
        println!("digraph G {{");
        for edge in edges.iter() {
            println!("    {} -> {};", edge.start_idx, edge.end_idx);
        }
        println!("}}");
    }
}

impl PartialOrd for UnfinishedEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.end_idx == other.start_idx {
            // || self.start_idx <= other.end_idx {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

#[derive(Clone, Default, Debug, Eq, PartialEq)]
struct TopologicalHalfEdgeMesh {
    /// All the vertices in the mesh
    edges: Vec<Edge<usize>>,
    edge_idxs: HashMap<(usize, usize), usize>,
    boundary_cycles: Vec<Vec<usize>>,
}

impl TopologicalHalfEdgeMesh {
    /// Used when building the mesh to add the edges of a triangle to the mesh if they're not
    /// already in there (they should not be in there since all edges are only contained in a
    /// single triangle)
    fn add_tri<'a>(
        triangle: [usize; 3],
        edges: &'a mut Vec<UnfinishedEdge>,
        edge_idxs: &'a mut HashMap<(usize, usize), usize>,
    ) -> &'a mut [UnfinishedEdge] {
        // println!("Adding tri");
        let mut inserted_edges_count = edges.len();
        let mut get_edge_idx = |start, stop| {
            let edge = (start, stop);
            *edge_idxs.entry(edge).or_insert_with(|| {
                inserted_edges_count += 1;
                inserted_edges_count - 1
            })
        };
        let mut interior: [MaybeUninit<UnfinishedEdge>; 3] = [
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
        ];
        let ptr_interior = [
            interior[0].as_mut_ptr(),
            interior[1].as_mut_ptr(),
            interior[2].as_mut_ptr(),
        ];
        let tri_edges = unsafe {
            for idx in 0..3 {
                let current_vertex = triangle[idx];
                let next_vertex = triangle[(idx + 1) % 3];
                let prev_vertex = triangle[(idx + 2) % 3];

                let _curr_edge_idx = get_edge_idx(current_vertex, next_vertex);
                let next_edge_idx = get_edge_idx(next_vertex, prev_vertex);
                let prev_edge_idx = get_edge_idx(prev_vertex, current_vertex);

                // set vertices
                addr_of_mut!((*ptr_interior[idx]).start_idx).write(current_vertex);
                addr_of_mut!((*ptr_interior[idx]).end_idx).write(next_vertex);

                // set next and prev
                addr_of_mut!((*ptr_interior[idx]).next_idx).write(Some(next_edge_idx));
                addr_of_mut!((*ptr_interior[idx]).prev_idx).write(Some(prev_edge_idx));

                // we start all edges off without a twin
                addr_of_mut!((*ptr_interior[idx]).twin_idx).write(None);
            }
            let [a, b, c] = interior;

            [a.assume_init(), b.assume_init(), c.assume_init()]
        };
        edges.extend(tri_edges);

        let n = edges.len();
        &mut edges.as_mut_slice()[n - 3..]
    }

    /// Consider chains like
    ///     ((a,b),(b,c),(c,d),(d,a))
    ///     ((e,f),(f,g),(g,h),(h,i),(i,e))
    /// [where all a,b,... unique integers] given as a bag of segments [2-tuples] in arbitrary order.
    /// We now want to create these chains from the bag.
    /// First off note that every integer occurs exactly once in the left and exactly once in the right position.
    /// This should be an invariant of our operation: every segment in the input should also occur exactly once in
    /// the output.
    /// Note that the integers in the left and right place being equal means that the sorted sequence of the left
    /// elements is identical to the sorted sequence of the right elements.
    /// We can thus match two segments of length 1 via their indices in the sorted sequences to produce chain-
    /// segments of length 2. However in doing so we gotta pay attention as to not duplicate any elements - we thus
    /// keep track of which segment has already been added and only add those that have not been added.
    /// Furthermore we note a chain is finished iff the leftmost and rightmost elements are equal.
    /// We apply this process recursively until all chains are finished / we've detected them to be cycles.
    ///
    /// A potentially easier to understand implementation is this python code:
    ///
    /// ```python3
    /// from typing import Iterable
    /// import numpy as np
    /// from typing import Iterable, Tuple
    ///
    /// Chain = Iterable[int]
    ///
    /// def build_chains(
    ///     segs: Iterable[Chain],
    ///     finished: Iterable[Chain]
    /// )-> Tuple[Iterable[Chain], Iterable[Chain]]:
    ///     if len(segs) == 0:
    ///         return segs, finished
    ///     if len(segs) == 1:
    ///         finished.extend(segs)
    ///         return [], finished
    ///     left_segs = sorted(segs, key=lambda x: x[-1])
    ///     right_segs = sorted(segs, key=lambda x: x[0])
    ///     idx_rl = np.argsort(np.array([x[0] for x in left_segs]))
    ///     ret = []
    ///     added = np.zeros(len(left_segs), dtype=np.bool8)
    ///     for i, (l_seg, r_seg) in enumerate(zip(left_segs, right_segs)):
    ///         if (not added[i]) and l_seg[0] == l_seg[-1]:
    ///             added[i] = True
    ///             finished.append(l_seg)
    ///         elif (not added[i]) and (not added[idx_rl[i]]):
    ///             added[i] = True
    ///             added[idx_rl[i]] = True
    ///             ret.append((*l_seg, *r_seg))
    ///     for seg, was_added in zip(left_segs, added):
    ///         if not was_added:
    ///             ret.append(seg)
    ///     return build_chains(ret, finished)
    ///
    ///
    /// xs = [(4, 0),
    ///     (0, 1),
    ///     (1, 2),
    ///     (2, 3),
    ///     (9, 4),
    ///     (6, 5),
    ///     (7, 6),
    ///     (11, 7),
    ///     (3, 8),
    ///     (13, 9),
    ///     (5, 10),
    ///     (10, 11),
    ///     (8, 12),
    ///     (14, 13),
    ///     (15, 14),
    ///     (12, 15)]
    ///
    /// print(build_chains(xs, []))
    /// ```
    fn find_boundary_cycles_helper<'a>(
        mut boundary: Vec<Vec<&'a mut UnfinishedEdge>>,
        mut finished_cycles: Vec<Vec<&'a mut UnfinishedEdge>>,
        edge_idxs: &HashMap<(usize, usize), usize>,
        n: usize,
    ) -> (
        Vec<Vec<&'a mut UnfinishedEdge>>,
        Vec<Vec<&'a mut UnfinishedEdge>>,
    ) {
        println!("{:?} ", n);
        match boundary.len() {
            0 => (boundary, finished_cycles),
            // while specialcasing this case would be slightly faster it'd require duplication of logic from below
            // to set prev and next on the first and last chain elements
            // 1 => {
            //     finished_cycles.extend(boundary);
            //     (vec![], finished_cycles)
            // }
            n_boundary_segments => {
                let mut left_sequence: Vec<_> = {
                    boundary.sort_by_key(|chain| chain.last().unwrap().end_idx);
                    boundary.into_iter().map(Some).collect()
                };
                let idxs_l = (0..n_boundary_segments).into_iter().collect::<Vec<_>>();
                let idxs_r = idxs_l.clone().into_iter().sorted_by_key(|i| {
                    left_sequence[*i]
                        .as_ref()
                        .map(|seg| seg.first().unwrap().start_idx)
                        .unwrap()
                });
                let mut ret = vec![];
                // let mut finished_cycles_idxs = vec![];
                for (i_left_seg, i_right_seg) in idxs_l.into_iter().zip(idxs_r) {
                    if let Some(mut left_seg) = left_sequence[i_left_seg].take() {
                        match {
                            let (head, tail) = left_seg.as_mut_slice().split_at_mut(1);
                            (head.first_mut(), tail.last_mut())
                        } {
                            (Some(left_start), Some(left_end)) => {
                                if left_start.start_idx == left_end.end_idx {
                                    let left_end_idx =
                                        edge_idxs[&(left_end.start_idx, left_end.end_idx)];
                                    left_start.prev_idx = Some(left_end_idx);
                                    let left_start_idx =
                                        edge_idxs[&(left_start.start_idx, left_start.end_idx)];
                                    left_end.next_idx = Some(left_start_idx);
                                    finished_cycles.push(left_seg);
                                } else if let Some(mut right_seg) =
                                    left_sequence[i_right_seg].take()
                                {
                                    let mut left_end = left_seg.last_mut().unwrap();
                                    let mut right_start = right_seg.first_mut().unwrap();

                                    left_end.next_idx = Some(
                                        edge_idxs[&(right_start.start_idx, right_start.end_idx)],
                                    );
                                    right_start.prev_idx =
                                        Some(edge_idxs[&(left_end.start_idx, left_end.end_idx)]);

                                    left_seg.extend(right_seg);
                                    ret.push(left_seg);
                                } else {
                                    // we put the segment back if we haven't used it
                                    left_sequence[i_left_seg] = Some(left_seg);
                                };
                            }
                            // there's only a single segment - the current chain has length 1
                            (Some(left_start), None) => {
                                if left_start.start_idx == left_start.end_idx {
                                    panic!("Kinda weird - boundary chain of length 1 doesn't make sense");
                                    // finished_cycles.push(left_seg);
                                } else if let Some(mut right_seg) =
                                    left_sequence[i_right_seg].take()
                                {
                                    let mut left_end = left_seg.last_mut().unwrap();
                                    let mut right_start = right_seg.first_mut().unwrap();

                                    left_end.next_idx = Some(
                                        edge_idxs[&(right_start.start_idx, right_start.end_idx)],
                                    );
                                    right_start.prev_idx =
                                        Some(edge_idxs[&(left_end.start_idx, left_end.end_idx)]);

                                    left_seg.extend(right_seg);
                                    ret.push(left_seg);
                                } else {
                                    // we put the segment back if we haven't used it
                                    left_sequence[i_left_seg] = Some(left_seg);
                                };
                            }
                            _ => panic!("ja shit"),
                        }
                    }
                }
                for seg in left_sequence.into_iter().flatten() {
                    ret.push(seg);
                }
                Self::find_boundary_cycles_helper(ret, finished_cycles, edge_idxs, n + 1)
            }
        }
    }

    fn find_boundary_cycles<'a>(
        boundary: Vec<&'a mut UnfinishedEdge>,
        edge_idxs: &HashMap<(usize, usize), usize>,
    ) -> Vec<Vec<usize>> {
        Self::find_boundary_cycles_helper(
            boundary.into_iter().map(|x| vec![x]).collect(),
            vec![],
            edge_idxs,
            0,
        )
        .1
        .into_iter()
        .map(|chain| {
            chain
                .into_iter()
                .map(|edge| edge_idxs[&(edge.start_idx, edge.end_idx)])
                .collect()
        })
        .collect()
    }

    fn construct_interior_mesh(
        root_triangle: [UnfinishedEdge; 3],
        edges: &mut Vec<UnfinishedEdge>,
        edge_idxs: &mut HashMap<(usize, usize), usize>,
        triangles: &mut Vec<[usize; 3]>,
    ) {
        for mut edge in root_triangle {
            if edge.twin_idx.is_some() {
                // println!("Twin already set");
                continue;
            } else {
                let twin_idxs = (edge.end_idx, edge.start_idx);
                if let Some(twin_idx) = edge_idxs.get(&twin_idxs) {
                    edge.twin_idx = Some(*twin_idx);
                    edges[*twin_idx].twin_idx =
                        edge_idxs.get(&(edge.start_idx, edge.end_idx)).copied();
                } else {
                    // println!("{:?}", twin_idxs);
                    // if the twin edge is part of some triangle
                    if let Some(triangle_idx) = triangles.iter().position(|triangle| {
                        triangle.iter().contains(&twin_idxs.0)
                            && triangle.iter().contains(&twin_idxs.1)
                    }) {
                        // TODO: this might be very inefficient. Maybe there's a better data structure than a Vec for this
                        let triangle = triangles.remove(triangle_idx);
                        // At this point the triangle is in some arbitrary order. We thus reorder it such that the edge
                        // we're currently processing aligns with its twin.
                        let triangle = [
                            twin_idxs.0,
                            twin_idxs.1,
                            triangle
                                .into_iter()
                                .find(|x| *x != twin_idxs.0 && *x != twin_idxs.1)
                                .unwrap(),
                        ];
                        // println!("Twin is in some triangle");
                        // We insert the triangle containing the twin edge and then set the twins for the dual (so the
                        // original twin) as well the primal edge
                        let primal_idx = edge_idxs[&(edge.start_idx, edge.end_idx)];
                        let twin_idx = edges.len();
                        // there is some triangle containing the twin edge, so we construct that triangle
                        let tri = Self::add_tri(triangle, edges, edge_idxs);
                        // for e in tri.iter() {
                        //     dbg!(e.prev_idx);
                        //     dbg!(e.next_idx);
                        // }
                        tri[0].twin_idx = Some(primal_idx);
                        edge.twin_idx = Some(twin_idx);
                        // construct the mesh continuing from the just generated triangle
                        Self::construct_interior_mesh(
                            tri.try_into().unwrap(),
                            edges,
                            edge_idxs,
                            triangles,
                        ); // trampoline this call?
                    } else {
                        // println!("Twin is in no triangle");
                        // there is no triangle containing the twin edge, so we construct it as a standalone edge
                        let twin = UnfinishedEdge {
                            start_idx: twin_idxs.0,
                            end_idx: twin_idxs.1,
                            prev_idx: None,
                            next_idx: None,
                            twin_idx: Some(edge_idxs[&(edge.start_idx, edge.end_idx)]),
                        };
                        // and insert it into the edges
                        edges.push(twin);
                        let twin_idx = edges.len() - 1;
                        edge_idxs.insert(twin_idxs, twin_idx);
                        edge.twin_idx = Some(twin_idx);
                    }
                }
            }
        }
    }
}

pub type TopologicalTriangle = [usize; 3];

impl TryFrom<Vec<TopologicalTriangle>> for TopologicalHalfEdgeMesh {
    type Error = ();
    /// Note that each half-edge is contained in at most one triangle. If it's not inside a
    /// triangle it's on the boundary.
    ///
    /// The basic construction works as follows:
    ///
    /// ```
    /// select some triangle
    /// insert interior half-edges and set their prev and next and is_on_boundary to False
    /// for inserted interior edge:
    ///     if twin edge is in another triangle T:
    ///         insert interior edges of T
    ///         set their prev and next
    ///         set twins on inserted interior edge and new edge
    ///     else:
    ///         insert twin edge and set is_on_boundary to True
    /// for edges on boundary:
    ///     set prev and next and split boundary into separate chains
    /// ```
    fn try_from(triangles: Vec<TopologicalTriangle>) -> Result<Self, Self::Error> {
        // We normalize the triangles by sorting them all - I'm not quite sure anymore why I did this
        let triangles: Vec<[usize; 3]> = triangles
            .into_iter()
            .map(|mut x| {
                x.sort();
                x
            })
            .collect();

        println!("Normalized / sorted");
        if let Some(triangle) = triangles.first() {
            // insert interior half-edges
            let mut edges = Vec::new();
            let mut edge_idxs = HashMap::new();
            let initial_triangle = Self::add_tri(*triangle, &mut edges, &mut edge_idxs);
            Self::construct_interior_mesh(
                initial_triangle.try_into().unwrap(),
                &mut edges,
                &mut edge_idxs,
                &mut triangles[1..].into(),
            );

            // at this point the full mesh has been constructed but edges on the boundary have no prev and next yet
            // we thus construct those next

            println!("Mesh constructed");
            let boundary_cycles = dbg!(Self::find_boundary_cycles(
                edges
                    .iter_mut()
                    .filter(|edge| edge.prev_idx.is_none() && edge.next_idx.is_none())
                    .collect::<Vec<_>>(),
                &edge_idxs,
            ));

            println!("Found boundary cycles");
            for edge in edges.clone().iter() {
                if let Some(idx) = edge.twin_idx {
                    edges[edge_idxs[&(edge.end_idx, edge.start_idx)]].twin_idx = Some(idx);
                }
            }

            println!("Set twins");
            // for cycle in boundary_cycles.iter() {
            //     println!("CYCLE:");
            //     for i in cycle.iter() {
            //         println!("{:?}", edges[*i]);
            //     }
            //     println!("\n");
            // }
            // println!("\n\n");
            // 'for_edge: for (i, edge) in edges.iter().enumerate() {
            //     for cycle in boundary_cycles.iter() {
            //         for j in cycle.iter() {
            //             if i == *j {
            //                 continue 'for_edge;
            //             }
            //         }
            //     }
            //     println!("{:?}", edge);
            // }

            Ok(TopologicalHalfEdgeMesh {
                edges: edges
                    .into_iter()
                    .map(|pre_edge| pre_edge.try_into())
                    .collect::<Result<_, _>>()?,
                edge_idxs,
                boundary_cycles,
            })
            // panic!("Done! :D");
        } else {
            // empty mesh
            Ok(TopologicalHalfEdgeMesh::default())
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GeometricHalfEdgeMesh<T> {
    /// All the vertices in the mesh
    vertices: Vec<Vertex<T>>,
    edges: Vec<Edge<T>>,
    edge_idxs: HashMap<(usize, usize), usize>,
    boundary_cycles: Vec<Vec<usize>>,
}

impl<T: Copy> GeometricHalfEdgeMesh<T> {}

fn main() {
    let raw_obj =
        std::fs::read_to_string("/home/stefan/GitHub/small-ray/resources/Stanford_Bunny.obj")
            .unwrap();
    let obj = wavefront_obj::obj::parse(raw_obj).unwrap();
    let object = obj.objects.first().unwrap();
    let vertices: Vec<Vertex<[f64; 3]>> = object
        .vertices
        .iter()
        .map(|wavefront_obj::obj::Vertex { x, y, z }| Vertex([*x, *y, *z]))
        .collect();
    let triangles = object
        .geometry
        .first()
        .unwrap()
        .shapes
        .iter()
        .filter_map(|shape| {
            if let wavefront_obj::obj::Primitive::Triangle((i1, _, _), (i2, _, _), (i3, _, _)) =
                shape.primitive
            {
                Some([i1, i2, i3])
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    // let vertices = vec![
    //     Vertex(1.0, 0.0, 0.0),
    //     Vertex(0.0, 1.0, 0.0),
    //     Vertex(0.0, 0.0, 1.0),
    // ];
    // let triangles = vec![[0, 1, 2]];

    //let vertices = vec![
    //    Vertex(-2, 0, 0),
    //    Vertex(0, 0, 0),
    //    Vertex(2, 0, 0),
    //    Vertex(-1, 2, 0),
    //    Vertex(1, 2, 0),
    //    Vertex(0, 4, 1),
    //];
    //let triangles = vec![[0, 1, 3], [1, 4, 2], [3, 4, 1], [5, 3, 4]];

    // let vertices = vec![];
    // let triangles = vec![
    //     // 16 triangles, 64 edges
    //     [7, 8, 3],
    //     [0, 1, 5],
    //     [0, 4, 5],
    //     [5, 6, 1],
    //     [1, 2, 6],
    //     [2, 7, 6],
    //     [2, 3, 7],
    //     //
    //     [4, 5, 9],
    //     [5, 9, 10],
    //     [7, 11, 12],
    //     [7, 8, 12],
    //     //
    //     [9, 10, 13],
    //     [10, 13, 14],
    //     [10, 11, 14],
    //     [11, 14, 15],
    //     [11, 12, 15],
    // ];
    println!("Got here");
    TopologicalHalfEdgeMesh::try_from(triangles).unwrap();
    dbg!(());
}
