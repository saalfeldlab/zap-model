#!/usr/bin/env python3
"""Verify that fishfuncem + neuprint connectivity works."""

from fishfuncem.em.NeuprintServer import NeuprintServer

server = NeuprintServer()
client = server.client
neuron_data = client.fetch_custom(
    """
    MATCH (n:Neuron)
    RETURN n.bodyId AS bid, n.type AS type, n.zapbenchId AS zb_id,
        n.statusLabel AS status_label,
        n.somaLocation.x AS soma_x,
        n.somaLocation.y AS soma_y,
        n.somaLocation.z AS soma_z
    LIMIT 5
    """,
    format="pandas",
)
print(neuron_data)
