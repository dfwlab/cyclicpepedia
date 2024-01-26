#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: graph_alignment.py
Author: Dingfeng Wu
Creator: Dingfeng Wu
Date Created: 2023-12-27
Last Modified: 2024-01-03
Version: 1.0.0
License: MIT License
Description: The alignment algorithm is developed by CyclicPepedia based on networkx extension, which can convert cyclic peptides into graphical structures and align them accordingly.

Copyright Information: Copyright (c) 2023 dfwlab (https://dfwlab.github.io/)

The code in this script can be used under the MIT License.
"""
import io
import re
import networkx as nx
from collections import Counter
from networkx.algorithms import isomorphism
from itertools import combinations
import matplotlib.pyplot as plt
from IPython.display import SVG

def sequence_to_node_edge(sequence):
    amino_acids = sequence.split('--')
    nodes = [] # 节点信息: [(0, 'Cys'), (1, 'Cys'), ...]
    edges = [] # 边信息: [(0, 1), (1, 2), ...]
    # 用于存储特殊连接信息
    special_connections = {}
    for i, amino_acid in enumerate(amino_acids):
        # 提取氨基酸名称和特殊连接信息
        amino_acid_name = re.sub(r"\(\d+\)", "", amino_acid)
        #print(amino_acid_name)
        if '-' in amino_acid_name:
            items = [k.strip() for k in amino_acid_name.split('-')]
            for j in range(len(items)-1):
                nodes.append((str(i)+'_'+str(j), items[j].strip().upper()))
                edges.append((str(i)+'_'+str(j), i))
            amino_acid_name = items[-1]
        nodes.append((i, amino_acid_name.strip().capitalize()))
        # 记录普通连接
        if i > 0:
            edges.append((i - 1, i))
        # 处理特殊连接
        special_conn_ids = re.findall(r"\((\d+)\)", amino_acid)
        #print(special_conn_ids)
        for conn_id in special_conn_ids:
            conn_id = int(conn_id) - 1  # 转换为从0开始的索引
            # 存储特殊连接信息
            if conn_id in special_connections:
                edges.append((special_connections[conn_id], i))
            else:
                special_connections[conn_id] = i
    return nodes, edges

def create_graph(nodes, edges):
    """根据给定的节点和边创建图"""
    G = nx.Graph()
    for node, code in nodes:
        G.add_node(node, code=code)
    G.add_edges_from(edges)
    return G

def plot_graph(G, dpi=60):
    plt.figure(figsize=(6, 6), dpi=dpi)
    pos = nx.spring_layout(G)  # 生成布局
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=800)
    labels = nx.get_node_attributes(G, 'code')
    # 调整标签位置
    label_pos = {k: [v[0], v[1] + 0] for k, v in pos.items()} 
    nx.draw_networkx_labels(G, label_pos, labels=labels)
    plt.show()

def node_match(n1, n2):
    """节点匹配函数，只有当节点的code属性相同时才匹配"""
    #print(n1['code'], n2['code'], n1['code'] == n2['code'])
    return n1['code'] == n2['code']

def generate_subgraphs_from_edges(query_G):
    """使用给定的边集生成子图，并找到所有连通分量"""
    subgraphs = {}
    for n in range(1, len(query_G.edges) + 1):
        for sub_edges in combinations(query_G.edges(), n):
            # Create a subgraph with the selected edges
            subg = query_G.edge_subgraph(sub_edges).copy()
            if nx.is_connected(subg) and subg.edges():
                subgraphs[len(subg.nodes)] = subgraphs.get(len(subg.nodes), [])
                subgraphs[len(subg.nodes)].append(subg)
    return subgraphs

def is_subgraph_of(query, ref):
    """检查 G1 是否是 G2 的子图"""
    GM = isomorphism.GraphMatcher(ref, query, node_match=node_match)
    return GM.subgraph_is_isomorphic()

def find_max_match_subgraph(subgraphs, ref_G):
    max_subg = None
    for n in sorted(subgraphs.keys()):
        #print(n, len(subgraphs[n]))
        for subg in subgraphs[n]:
            if is_subgraph_of(subg, ref_G):
                max_subg = subg
                break
        else:
            break
    return n, subg, max_subg

def mcs2similarity(ref, mcs):
    return float(len(mcs.nodes)+len(mcs.edges))/(len(ref.nodes)+len(ref.edges))

def mcs_similarity(query, reference):
    subgraphs = generate_subgraphs_from_edges(query)
    n, subg, max_subg = find_max_match_subgraph(subgraphs, reference)
    if max_subg:
        return len(max_subg.nodes)+len(max_subg.edges), mcs2similarity(reference, max_subg)
    return 0, 0.0
    
def graph_similarity(query, reference):
    distance = nx.graph_edit_distance(query, reference, node_match=node_match)
    similarity = 1-distance/float(len(query.nodes)+len(query.edges)+len(reference.nodes)+len(reference.edges))
    return distance, similarity

def graph2svg(G, dpi=60):
    plt.figure(figsize=(6, 6), dpi=dpi)
    pos = nx.spring_layout(G)  # 生成布局
    svg_buffer = io.StringIO()
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=800)
    labels = nx.get_node_attributes(G, 'code')
    # 调整标签位置
    label_pos = {k: [v[0], v[1] + 0] for k, v in pos.items()} 
    nx.draw_networkx_labels(G, label_pos, labels=labels)
    plt.savefig(svg_buffer, format="svg", bbox_inches='tight')
    plt.close()
    svg_text = svg_buffer.getvalue()
    return svg_text

def amino_acid_composition_freq(query, ref):
    query_freq = Counter(query)
    ref_freq = Counter(ref)
    query_freq_norm = {aa: count / float(len(query_freq)) for aa, count in query_freq.items()}
    ref_freq_norm = {aa: count / float(len(ref_freq)) for aa, count in ref_freq.items()}
    all_amino_acids = set(query_freq_norm.keys()).union(set(ref_freq_norm.keys()))
    # Calculate similarity
    min_similarity = sum(min(query_freq_norm.get(aa, 0), ref_freq_norm.get(aa, 0)) for aa in all_amino_acids)
    max_similarity = sum(max(query_freq_norm.get(aa, 0), ref_freq_norm.get(aa, 0)) for aa in all_amino_acids)
    similarity = min_similarity / max_similarity if max_similarity > 0 else 0
    return similarity


