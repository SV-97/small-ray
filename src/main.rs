#![allow(dead_code)]

// mod trampoline;

mod mesh;
use mesh::{topological::TopologicalHalfEdgeMesh, Vertex};

fn main() {
    let path = "/home/stefan/GitHub/small-ray/resources/\
        Stanford_Bunny.obj";
    // single-wireless-stereo-speakers-3d-model/Single Wireless Stereo Speakers 3D Model.obj";
    let raw_obj = std::fs::read_to_string(path).unwrap();
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
    dbg!(triangles.len());
    dbg!(vertices.len());
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
    dbg!(topological::TopologicalHalfEdgeMesh::try_from(triangles).unwrap());
}
