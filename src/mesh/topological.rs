use std::{collections::HashMap, mem::MaybeUninit, ptr::addr_of_mut};

use itertools::Itertools;

use super::{Edge, UnfinishedEdge};

/// A half-edge mesh based solely on the topology of some points
#[derive(Clone, Default, Debug, Eq, PartialEq)]
pub struct TopologicalHalfEdgeMesh {
    /// All the edges in the mesh
    edges: Vec<Edge<usize>>,
    /// Maps pairs `(start_idx, end_idx)` into the corresponding half-edge via indices into `edges`
    edge_idxs: HashMap<(usize, usize), usize>,
    /// For every boundary cycle this contains a collection of indices into edges. So a meshed cylinder with open
    /// top and bottom would contain one cycle for the top "ring" and one for the bottom one.
    boundary_cycles: Vec<Vec<usize>>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ConstructionTrampoline {
    Done,
    Thunk([UnfinishedEdge; 3]),
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
    ) -> (
        Vec<Vec<&'a mut UnfinishedEdge>>,
        Vec<Vec<&'a mut UnfinishedEdge>>,
    ) {
        match boundary.len() {
            0 => (boundary, finished_cycles),
            // while specialcasing this case would be slightly faster it'd require duplication of logic from below
            // to set prev and next on the first and last chain elements
            // 1 => {
            //     finished_cycles.extend(boundary);
            //     (vec![], finished_cycles)
            // }
            n_boundary_segments => {
                // we map with Some so that we're able to take elements out and put them back. If this becomes
                // a problem performancewise we may omit the checks by using `MaybeUninit` and unsafe rather than
                // `Option`.
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
                            _ => unreachable!(),
                        }
                    }
                }
                for seg in left_sequence.into_iter().flatten() {
                    ret.push(seg);
                }
                Self::find_boundary_cycles_helper(ret, finished_cycles, edge_idxs)
            }
        }
    }

    /// Given a collection of unfinished edges with only start/end set this finds all "cycles" ( so closed chains
    /// of edges such that start and end of two successive edges are matched and the next of the last one is the
    /// first one / prev of first one is last one ), setting next/prev in the process
    fn find_boundary_cycles<'a>(
        boundary: Vec<&'a mut UnfinishedEdge>,
        edge_idxs: &HashMap<(usize, usize), usize>,
    ) -> Vec<Vec<usize>> {
        Self::find_boundary_cycles_helper(
            boundary.into_iter().map(|x| vec![x]).collect(),
            vec![],
            edge_idxs,
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

    fn construct_interior_mesh_from_edge(
        mut root_edge: UnfinishedEdge,
        edges: &mut Vec<UnfinishedEdge>,
        edge_idxs: &mut HashMap<(usize, usize), usize>,
        triangles: &mut Vec<[usize; 3]>,
    ) -> ConstructionTrampoline {
        let twin_idxs = (root_edge.end_idx, root_edge.start_idx);
        if let Some(twin_idx) = edge_idxs.get(&twin_idxs) {
            root_edge.twin_idx = Some(*twin_idx);
            edges[*twin_idx].twin_idx = edge_idxs
                .get(&(root_edge.start_idx, root_edge.end_idx))
                .copied();
        } else {
            // println!("{:?}", twin_idxs);
            // if the twin edge is part of some triangle
            if let Some(triangle_idx) = triangles.iter().position(|triangle| {
                triangle.iter().contains(&twin_idxs.0) && triangle.iter().contains(&twin_idxs.1)
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
                let primal_idx = edge_idxs[&(root_edge.start_idx, root_edge.end_idx)];
                let twin_idx = edges.len();
                // there is some triangle containing the twin edge, so we construct that triangle
                let tri = Self::add_tri(triangle, edges, edge_idxs);
                // the first edge in the just generated triangle is the one corresponding to our root edge
                tri[0].twin_idx = Some(primal_idx);
                root_edge.twin_idx = Some(twin_idx);
                // construct the mesh continuing from the just generated triangle
                return ConstructionTrampoline::Thunk(tri.try_into().unwrap());
            } else {
                // println!("Twin is in no triangle");
                // there is no triangle containing the twin edge, so we construct it as a standalone edge
                let twin = UnfinishedEdge {
                    start_idx: twin_idxs.0,
                    end_idx: twin_idxs.1,
                    prev_idx: None,
                    next_idx: None,
                    twin_idx: Some(edge_idxs[&(root_edge.start_idx, root_edge.end_idx)]),
                };
                // and insert it into the edges
                edges.push(twin);
                let twin_idx = edges.len() - 1;
                edge_idxs.insert(twin_idxs, twin_idx);
                root_edge.twin_idx = Some(twin_idx);
            }
        }
        ConstructionTrampoline::Done
    }

    /// Construct a mesh given by `triangles` by "discovering" it starting at some `root_triangle`. In the process
    /// it'll write new edges into `edges` writing their indices inside of `edges` into `edge_idxs` under the
    /// key `(start_idx, end_idx)`.
    fn construct_interior_mesh(
        root_triangle: [UnfinishedEdge; 3],
        edges: &mut Vec<UnfinishedEdge>,
        edge_idxs: &mut HashMap<(usize, usize), usize>,
        triangles: &mut Vec<[usize; 3]>,
    ) {
        /*
        This uses trampolining (not quite but pretty close) to enable processing of large meshes via `construct_interior_mesh_from_edge`.
        Each call may produce up to three new units of work - which we then process one after another.
        */
        let mut thunks_to_process = vec![root_triangle];
        while let Some(tri) = thunks_to_process.pop() {
            for edge in tri {
                if edge.twin_idx.is_some() {
                    // println!("Twin already set");
                    continue;
                } else {
                    match Self::construct_interior_mesh_from_edge(edge, edges, edge_idxs, triangles)
                    {
                        ConstructionTrampoline::Thunk(tri) => thunks_to_process.push(tri),
                        ConstructionTrampoline::Done => (),
                    }
                }
            }
        }
    }
}

/// A topological triangle is just a bunch (3) of abstract points that are mutually connected.
/// To refer to these points we consider them as parts of some enumerable collection of points
/// and identify them with their index.
pub type TopologicalTriangle = [usize; 3];

impl TryFrom<Vec<TopologicalTriangle>> for TopologicalHalfEdgeMesh {
    type Error = ();
    /// Generates a mesh from the given triangle data - note that this will fail for non-connected
    /// and non-manifold meshes.
    ///
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

            let boundary_cycles = Self::find_boundary_cycles(
                edges
                    .iter_mut()
                    .filter(|edge| edge.prev_idx.is_none() && edge.next_idx.is_none())
                    .collect::<Vec<_>>(),
                &edge_idxs,
            );

            for edge in edges.clone().iter() {
                if let Some(idx) = edge.twin_idx {
                    edges[edge_idxs[&(edge.end_idx, edge.start_idx)]].twin_idx = Some(idx);
                }
            }

            Ok(TopologicalHalfEdgeMesh {
                edges: edges
                    .into_iter()
                    .map(|pre_edge| pre_edge.try_into())
                    .collect::<Result<_, _>>()?,
                edge_idxs,
                boundary_cycles,
            })
        } else {
            // empty mesh
            Ok(TopologicalHalfEdgeMesh::default())
        }
    }
}
