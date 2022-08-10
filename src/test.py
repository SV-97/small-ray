from typing import TypeVar, FrozenSet, Generic
from dataclasses import dataclass
from functools import lru_cache

T = TypeVar("T")


@dataclass
class Graph(Generic[T]):
    vertices: FrozenSet[T]
    edges: FrozenSet[FrozenSet[T]]

    def to_dot(self) -> str:
        return "\n".join([
            "graph G {",
            ";\n".join(f'    "{to_tup(v)}"' for v in self.vertices),
            ";\n".join(f'    "{(t := to_tup(edge))[0]}" -- "{t[1]}"' for edge in self.edges),
            "}",
        ])


@lru_cache
def to_tup(xs):
    if isinstance(xs, frozenset):
        return tuple([to_tup(x) for x in xs])
    else:
        return xs


def vert_to_edge(graph: Graph[T]) -> Graph[FrozenSet[T]]:
    new_vertices = graph.edges
    new_edges = frozenset({frozenset({edge_1, edge_2})
        for edge_1 in new_vertices
        for edge_2 in new_vertices
        if edge_1 != edge_2 and any(vertex in edge_2 for vertex in edge_1)
    })
    return Graph(new_vertices, new_edges)

g0 = Graph(
    frozenset(range(1,6)),
    frozenset([
        frozenset({1,5}),
        frozenset({1,4}),
        frozenset({1,3}),
        frozenset({1,2}),
        frozenset({3,4}),
        frozenset({4,5}),
    ])
)

print(g0.to_dot())
gn = g0
for n in range(4):
    gn = vert_to_edge(gn)
    print(gn.to_dot())
