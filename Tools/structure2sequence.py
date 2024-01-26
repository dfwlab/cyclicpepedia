#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: structure2sequence.py
Author: Dingfeng Wu
Creator: Dingfeng Wu
Date Created: 2022-11-15
Last Modified: 2023-12-26
Version: 1.0.1
License: MIT License
Description: Structure-to-Sequence (Struc2seq) is a computing process based on RDkit and the characteristics of cyclic peptide sequences, which can convert cyclic peptide SMILES into sequence information.

Copyright Information: Copyright (c) 2023 dfwlab (https://dfwlab.github.io/)

The code in this script can be used under the MIT License.
"""

from io import BytesIO
import re
import numpy as np
import pandas as pd
from itertools import product

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, RWMol
from rdkit.Chem.Draw import rdMolDraw2D
import rdkit.Chem.rdmolfiles  as rdmol
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import SVG

def mol2seq_for_essentialAA(m):
    aa_smiles = {'ALA': 'C[C@H](N)C=O', 'CYS': 'N[C@H](C=O)CS', 'ASP': 'N[C@H](C=O)CC(=O)O', 'GLU': 'N[C@H](C=O)CCC(=O)O', 'PHE': 'N[C@H](C=O)Cc1ccccc1', 'GLY': 'NCC=O', 'HIS': 'N[C@H](C=O)Cc1c[nH]cn1', 'ILE': 'CC[C@H](C)[C@H](N)C=O', 'LYS': 'NCCCC[C@H](N)C=O', 'LEU': 'CC(C)C[C@H](N)C=O', 'MET': 'CSCC[C@H](N)C=O', 'ASN': 'NC(=O)C[C@H](N)C=O', 'PRO': 'O=C[C@@H]1CCCN1', 'GLN': 'NC(=O)CC[C@H](N)C=O', 'ARG': 'N=C(N)NCCC[C@H](N)C=O', 'SER': 'N[C@H](C=O)CO', 'THR': 'C[C@@H](O)[C@H](N)C=O', 'VAL': 'CC(C)[C@H](N)C=O', 'TRP': 'N[C@H](C=O)Cc1c[nH]c2ccccc12','TYR': 'N[C@H](C=O)Cc1ccc(O)cc1'}
    aas = ['GLY','ALA', 'VAL', 'CYS', 'ASP', 'GLU', 'PHE', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'TRP','TYR'] #order important because gly is substructure of other aas
    # detect the atoms of the backbone and assign them with info
    CAatoms = m.GetSubstructMatches(Chem.MolFromSmarts("[C:0](=[O:1])[C:2][N:3]"))
    #print(CAatoms)
    for atoms in CAatoms:
        a = m.GetAtomWithIdx(atoms[2])
        info = Chem.AtomPDBResidueInfo()
        info.SetName(" CA ") #spaces are important
        a.SetMonomerInfo(info)
    # detect the presence of residues and set residue name for CA atoms only
    for curr_aa in aas:
        matches = m.GetSubstructMatches(Chem.MolFromSmiles(aa_smiles[curr_aa]))
        for atoms in matches:
            for atom in atoms:
                a = m.GetAtomWithIdx(atom)
                info = Chem.AtomPDBResidueInfo()
                if a.GetMonomerInfo() != None:
                    if a.GetMonomerInfo().GetName() == " CA ":
                        info.SetName(" CA ")
                        info.SetResidueName(curr_aa)
                        a.SetMonomerInfo(info)
    # renumber the backbone atoms so the sequence order is correct:
    # N端乙酰化和C端酰胺化: https://zhuanlan.zhihu.com/p/540769025
    # 肽链一般从N端读到C端,C端一般保留一个COOH,因此C端为HO-C(=O),中间一个C,然后加一个肽键NC(=O),后续就都是C和肽键，一直到N端
    # 正反识别都可以，反向就是OC(=O)CNC(=O)CNC(=O)CN...,如果C端酰胺化,则为NC(=O)CNC(=O)CNC(=O)CN...
    # N端不管有没有乙酰化,都会保留N,无乙酰化为-NH2,有乙酰化为-NC(=O)CH3;
    bbsmiles = "C(=O)CN"*len(m.GetSubstructMatches(Chem.MolFromSmiles(aa_smiles["GLY"]))) # generate backbone SMILES
    backbone = m.GetSubstructMatches(Chem.MolFromSmiles(bbsmiles))[0]
    if not len(backbone):
        return "Not AA"
    id_list = list(backbone)
    id_list.reverse()
    #print(id_list)
    for idx in [a.GetIdx() for a in m.GetAtoms()]:
        if idx not in id_list:
            id_list.append(idx)
    #print(id_list)
    m_renum = Chem.RenumberAtoms(m, newOrder=id_list)
    return m_renum, Chem.MolToSequence(m_renum)

def blend_colors(c1, c2, alpha):
    # 解包颜色和透明度
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    # 计算混合后的颜色
    r = r1 * alpha + r2 * alpha * (1 - alpha)
    g = g1 * alpha + g2 * alpha * (1 - alpha)
    b = b1 * alpha + b2 * alpha * (1 - alpha)
    return (r, g, b)

def plot_smiles(smiles, w=600, h=600):
    try:
        m = Chem.MolFromSmiles(smiles)
    except:
        return ''
    # 设置绘图选项，调整分子的大小
    d2d = rdMolDraw2D.MolDraw2DSVG(w, h)
    
    # 绘制分子
    d2d.DrawMolecule(m)
    d2d.FinishDrawing()
    # 显示图像
    svg = d2d.GetDrawingText()
    return svg

def highlight_atom(m, atoms, transparency=1, isdisplay=True):
    # 设置绘图选项，调整分子的大小
    d2d = rdMolDraw2D.MolDraw2DSVG(600, 600)
    d2d.drawOptions().addAtomIndices = True
    d2d.drawOptions().addStereoAnnotation = True
    d2d.drawOptions().bondLineWidth = 2
    d2d.drawOptions().minFontSize = 15 # 调整字体大小
    d2d.drawOptions().annotationFontScale = 0.7 # 调整编号大小比例
    d2d.drawOptions().atomHighlightsAreCircles = True
    # 设置高亮颜色和大小
    highlight_colors = {}
    cmap = cm.get_cmap('tab20', len(atoms))
    for i, group in enumerate(atoms):
        for atom in group:
            if atom in highlight_colors.keys():
                highlight_colors[atom] = blend_colors(cmap(i)[:3], highlight_colors[atom], transparency) 
            else:
                highlight_colors[atom] = cmap(i)[:3] # 取RGB颜色，忽略Alpha
    highlight_radii = {atom: 1 for group in atoms for atom in group}
    # 绘制分子
    d2d.DrawMolecule(m, highlightAtoms=[atom for group in atoms for atom in group], 
                     highlightAtomColors=highlight_colors, highlightAtomRadii=highlight_radii)
    d2d.FinishDrawing()
    # 显示图像
    svg = d2d.GetDrawingText()
    #svg = re.sub(r"style='fill:([^;]+);", r"style='fill:\1;fill-opacity:"+str(transparency)+";", svg)
    if isdisplay:
        display(SVG(svg.replace('svg:', '')))
    return svg

def detect_backbone(m):
    ### 识别骨架并将骨架分子编号为0至N，其他侧链分子编号大于N
    # renumber the backbone atoms so the sequence order is correct:
    # N端乙酰化和C端酰胺化: https://zhuanlan.zhihu.com/p/540769025
    # 肽链一般从N端读到C端,C端一般保留一个COOH,因此C端为HO-C(=O),中间一个C,然后加一个肽键NC(=O),后续就都是C和肽键，一直到N端
    # 正反识别都可以，反向就是OC(=O)CNC(=O)CNC(=O)CN...,如果C端酰胺化,则为NC(=O)CNC(=O)CNC(=O)CN...
    # N端不管有没有乙酰化,都会保留N,无乙酰化为-NH2,有乙酰化为-NC(=O)CH3;
    bbsmiles = "C(=O)CN"*len(m.GetSubstructMatches(Chem.MolFromSmiles('NCC=O'))) # 使用GLY最小氨基酸单元识别氨基酸数量，生成骨架generate backbone SMILES
    backbone = m.GetSubstructMatches(Chem.MolFromSmiles(bbsmiles))[0]
    if not len(backbone):
        return False # 如果找不到肽键，或者数量不同则退出，比如存在两个线性肽链通过两个链接形成环的情况，需要逐一搜索；
    backbone_idx = list(backbone)
    backbone_idx.reverse()
    return backbone_idx

def order_backbone(m, backbone_idx):
    id_list = backbone_idx[:]
    # 重新编号
    for idx in [a.GetIdx() for a in m.GetAtoms()]:
        if idx not in id_list:id_list.append(idx)
    m_renum = Chem.RenumberAtoms(m, newOrder=id_list)
    # 重新找骨架
    backbone_idx = detect_backbone(m_renum)
    return m_renum, backbone_idx

def side_chain_neighbor(m, atoms, origin_aa_idx):
    neighbor_atoms = set()
    for atom_idx in atoms:
        atom = m.GetAtomWithIdx(atom_idx)
        # 遍历这个原子的邻居原子
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in origin_aa_idx:
                neighbor_atoms.add(neighbor_idx)
    return [i for i in neighbor_atoms if i not in atoms]

def aa_side_chain_extend(m, aa_set):
    # 会有重叠，主要发生在支链成环位置，两个氨基酸会都扩展并覆盖成环支链
    origin_aa_idx = []
    for i in aa_set:
        origin_aa_idx.extend(i[:])
    expanded_aa_set = []
    for aa in [set(i[:]) for i in aa_set]:
        neighbor_atoms = side_chain_neighbor(m, aa, origin_aa_idx)
        aa.update(neighbor_atoms)
        while(len(neighbor_atoms)):
            neighbor_atoms = side_chain_neighbor(m, aa, origin_aa_idx)
            aa.update(neighbor_atoms)
        expanded_aa_set.append(list(aa))
    return expanded_aa_set

def split_aa_unit(m):
    # 每个氨基酸定位，使用GLY最小氨基酸单元进行识别
    aa_units = m.GetSubstructMatches(Chem.MolFromSmiles('NCC=O'))
    #print(aa_units)
    # 支链扩张，从GLY结构扩张每个氨基酸支链，获取每个氨基酸结构
    expanded_aa_set = aa_side_chain_extend(m, aa_units)
    return expanded_aa_set
    
# 氨基酸标注
def get_complete_aas(m, aa_units):
    aas = []
    atom_mappings = []
    for atom_idxs in aa_units:
        # 创建一个新的可写分子对象
        new_mol = RWMol()
        # 添加原子并创建索引映射
        atom_mapping = {}
        for idx in atom_idxs:
            atom = m.GetAtomWithIdx(idx)
            new_idx = new_mol.AddAtom(atom)
            atom_mapping[idx] = new_idx
        atom_mappings.append(atom_mapping)
        # 添加键并处理外部连接
        for bond in m.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx in atom_idxs and end_idx in atom_idxs:
                # 如果两个原子都在列表中，添加键
                new_mol.AddBond(atom_mapping[begin_idx], atom_mapping[end_idx], bond.GetBondType())
            elif begin_idx in atom_idxs or end_idx in atom_idxs:# 如果其中一个原子在列表中，另一个不在
                idx = begin_idx if begin_idx in atom_idxs else end_idx
                atom = m.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    # 检查碳原子是否连接一个氧原子和一个氮原子（肽键）
                    has_double_bonded_oxygen = False
                    has_single_bonded_nitrogen = False
                    for neighbor in atom.GetNeighbors():
                        # 检查邻居原子是否为氧原子，并且键类型为双键
                        if neighbor.GetSymbol() == 'O' and m.GetBondBetweenAtoms(idx, neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                            has_double_bonded_oxygen = True
                        elif neighbor.GetSymbol() == 'N' and m.GetBondBetweenAtoms(idx, neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                            has_single_bonded_nitrogen = True
                    if has_double_bonded_oxygen and has_single_bonded_nitrogen:
                        # 在碳原子上添加一个 -OH 集团
                        new_oxygen_idx = new_mol.AddAtom(Chem.Atom('O'))
                        new_mol.AddBond(atom_mapping[idx], new_oxygen_idx, Chem.BondType.SINGLE)
                        #new_hydrogen_idx = new_mol.AddAtom(Chem.Atom('H'))
                        #new_mol.AddBond(new_oxygen_idx, new_hydrogen_idx, Chem.BondType.SINGLE)
                else:
                    pass
                    # 如果其中一个原子在列表中，另一个不在，添加虚原子
                    #inner_idx = begin_idx if begin_idx in atom_idxs else end_idx
                    #new_idx = new_mol.AddAtom(Chem.Atom(1))  # 添加氢原子作为占位符
                    #new_mol.AddBond(atom_mapping[inner_idx], new_idx, bond.GetBondType())
        new_mol = new_mol.GetMol()
        Chem.SanitizeMol(new_mol)
        aas.append(new_mol)
    return aas, atom_mappings

def reference_aa_monomer(path='monomer.tsv'):
    monomers = pd.read_csv(path, sep='\t', index_col=0)
    essentials = {}
    others = {}
    temp = monomers.loc[monomers['Essential amino acids']==1, :]
    for i in temp.index:
        smile = temp.loc[i, 'Smiles']
        code = temp.loc[i, 'Code']
        weight = float(temp.loc[i, 'Weight'])
        symbol = temp.loc[i, 'Symbol']
        symbol = code if symbol.strip()=='' or str(symbol).upper().strip()=='NAN' else symbol
        aa = Chem.MolFromSmiles(smile)
        num_atoms = aa.GetNumAtoms()
        essentials[code] = [aa, symbol, num_atoms]
    temp = monomers.loc[(monomers['Essential amino acids']!=1)&monomers['Error']!=1, :]
    for i in temp.index:
        smile = temp.loc[i, 'Smiles']
        code = temp.loc[i, 'Code']
        weight = float(temp.loc[i, 'Weight'])
        symbol = temp.loc[i, 'Symbol']
        symbol = code if str(symbol).strip()=='' or str(symbol).upper().strip()=='NAN' else symbol
        aa = Chem.MolFromSmiles(smile)
        num_atoms = aa.GetNumAtoms()
        others[code] = [aa, symbol, num_atoms]
    return essentials, others
    
def get_connected_pairs(m, atom_mappings):
    aa_idxs = [i.keys() for i in atom_mappings]
    overlap_aa_pairs = []
    #for i in range(len(aa_idxs)-1):
    #    for j in range(i+1, len(aa_idxs)):
    #        if len(set(aa_idxs[i])&set(aa_idxs[j])):
    #            overlap_aa_pairs.append((i, j, 'side chain'))
    # 确认主链首尾是否相连
    for aa_i in range(len(aa_idxs)-1):
        for aa_j in range(aa_i+1, len(aa_idxs)):
            first_aa_idx = aa_idxs[aa_i]
            last_aa_idx = aa_idxs[aa_j]
            is_link = False
            for bond in m.GetBonds():
                begin_idx, end_idx = sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                if (begin_idx in first_aa_idx and end_idx in last_aa_idx) or (end_idx in first_aa_idx and begin_idx in last_aa_idx):
                    is_link = True
                    is_peptide_bond = False
                    if m.GetAtomWithIdx(begin_idx).GetSymbol() == 'N' and m.GetAtomWithIdx(end_idx).GetSymbol() == 'C':
                        for neighbor in m.GetAtomWithIdx(end_idx).GetNeighbors():
                            if neighbor.GetSymbol() == 'O' and m.GetBondBetweenAtoms(end_idx, neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                                is_peptide_bond = True
                                break
                    elif m.GetAtomWithIdx(end_idx).GetSymbol() == 'N' and m.GetAtomWithIdx(begin_idx).GetSymbol() == 'C':
                        for neighbor in m.GetAtomWithIdx(begin_idx).GetNeighbors():
                            if neighbor.GetSymbol() == 'O' and m.GetBondBetweenAtoms(begin_idx, neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                                is_peptide_bond = True
                                break
            if is_link and is_peptide_bond:
                overlap_aa_pairs.append((aa_i, aa_j, 'peptide bond'))
            elif is_link:
                overlap_aa_pairs.append((aa_i, aa_j, 'side chain'))
            
    return overlap_aa_pairs

def search_one_chain(query_idx, search_idxs, N, connected_pairs):
    r = 0
    chain = [query_idx]
    while(r < N):
        for i, j, t in connected_pairs:
            if i == query_idx and j in search_idxs and t!='side chain':
                query_idx = j
                chain.append(query_idx)
                search_idxs = [k for k in search_idxs if k not in chain]
                break
            elif j == query_idx and i in search_idxs and t!='side chain':
                query_idx = i
                chain.append(query_idx)
                search_idxs = [k for k in search_idxs if k not in chain]
                break
        r += 1
    return chain

def get_connected_chain(connected_pairs, N_aa):
    search_res = []
    for query_idx in range(N_aa):
        search_idxs = [i for i in range(N_aa) if i != query_idx]
        chain = search_one_chain(query_idx, search_idxs, len(search_idxs), connected_pairs)
        search_res.append(chain)
    chain = sorted(search_res, key=lambda x:len(x), reverse=True)[0]
    if len(chain) == N_aa:
        return chain
    cn = 0
    while(cn < 5): # 最大5条单独链
        if len(chain)>N_aa-2:
            break
        search_res = []
        for query_idx in [k for k in range(N_aa) if k not in chain]:
            search_idxs = [i for i in range(N_aa) if i != query_idx and i not in chain]
            temp = search_one_chain(query_idx, search_idxs, len(search_idxs), connected_pairs)
            search_res.append(temp)
        chain += sorted(search_res, key=lambda x:len(x), reverse=True)[0]
        cn += 1
    if len(chain) < N_aa:
        chain += [i for i in range(N_aa) if i not in chain]
    return chain

def aa_matching(query_aas, atom_mappings, connected_pairs, ref_essentials, ref_others):
    result = []
    # 匹配氨基酸参考库
    for aa_i in range(len(query_aas)):
        aa = query_aas[aa_i]
        aa_num_atoms = aa.GetNumAtoms()
        for ref_code, [ref, ref_symbol, ref_num_atoms] in ref_essentials.items():
            if aa_num_atoms != ref_num_atoms:
                continue
            match_idx = aa.GetSubstructMatches(ref)
            if match_idx and len(match_idx[0])==aa_num_atoms:
                # 结果:[code, query氨基酸atom数量, 是否essential, 匹配原子数量, 匹配原子原始idx, 匹配原子在原分子的idx, 是否完全匹配]
                result.append([[ref_code, aa_num_atoms, True, len(match_idx[0]), match_idx[0], [], True]])
                break
        else:
            refs = {**ref_essentials, **ref_others}
            refs = {key: value for key, value in refs.items() if value[2]<=aa_num_atoms}
            match_codes = []
            for ref_code, [ref, ref_symbol, ref_num_atoms] in refs.items():
                match_idx = aa.GetSubstructMatches(ref)
                if match_idx:
                    match_idx = sorted(match_idx, key=lambda x:len(x), reverse=True)[0]
                    match_codes.append([ref_code, aa_num_atoms, False, len(match_idx), match_idx, [], True if aa_num_atoms==len(match_idx) else False, False])
            result.append(match_codes)
    # 删除异构体，比如同时对到Cys和D-Cys, 删除D-Cys，保留Cys; 同时确定原子映射
    for res_i in range(len(result)):
        atom_mapping = atom_mappings[res_i]
        atom_mapping = {value: key for key, value in atom_mapping.items()}
        res = result[res_i]
        new_res = []
        for match_i in res:
            for match_j in res:
                if match_i[0]!=match_j[0] and match_j[0] in match_i[0] and match_j[3]==match_i[3]:
                    break
            else:
                match_idx = tuple([atom_mapping[k] for k in match_i[4] if k in atom_mapping.keys()])
                match_i[5] = match_idx
                new_res.append(match_i)
        result[res_i] = new_res
    # 冲突解决：对于有交叉的氨基酸，确定最长且不交叉的的组合
    # 1. 交叉氨基酸确定
    overlap_aa_idx = set()
    for i, j, t in connected_pairs:
        if t == 'side chain':
            overlap_aa_idx.update([i, j])
    # 2. 非交叉氨基酸选择最大匹配，交叉氨基酸考虑两个氨基酸的匹配不交叉，并覆盖最大数量原子
    chain_aas = {}
    for aa_i in range(len(result)):
        if aa_i not in overlap_aa_idx:
            max_match_atom = 0
            max_matchs = []
            for match in result[aa_i]:
                if match[3]>max_match_atom:
                    max_match_atom = match[3]
                    max_matchs = [match]
                elif match[3]==max_match_atom:
                    max_matchs.append(match)
            chain_aas[(aa_i, )] = max_matchs
    for aa_i, aa_j, t in connected_pairs:
        if t == 'peptide bond':
            continue
        pairs = []
        for match_i in result[aa_i]:
            for match_j in result[aa_j]:
                if len(set(match_i[5])&set(match_j[5]))==0:
                    pairs.append([match_i, match_j, len(list(match_i[4])+list(match_j[4]))])
        max_atoms = max([i[2] for i in pairs])
        pairs = [i for i in pairs if i[2]>=max_atoms]
        chain_aas[(aa_i, aa_j)] = pairs
    # 3. 组合生成链
    chain_aas = list(chain_aas.items())
    combination_idx = [list(range(len(i[1]))) for i in chain_aas]
    combinations = list(product(*combination_idx))
    chains = []
    for combination in combinations:
        chain = [[] for i in range(len(query_aas))]
        for i in range(len(chain_aas)):
            idx = chain_aas[i][0]
            values = chain_aas[i][1]
            if len(idx)==1:
                chain[idx[0]] = values[combination[i]]
            else:
                chain[idx[0]] = values[combination[i]][0]
                chain[idx[1]] = values[combination[i]][1]
        chains.append(chain)
    return chains   

def reorder_result(order_chain, aas, aa_units, chains, connected_pairs):
    id_map = {order_chain[i]:i for i in range(len(order_chain))}
    new_aas = [None for i in range(len(aas))]
    for i in range(len(aas)):
        new_aas[id_map[i]] = aas[i]
    new_aa_units = [None for i in range(len(aa_units))]
    for i in range(len(aa_units)):
        new_aa_units[id_map[i]] = aa_units[i]
    new_chains = []
    for chain in chains:
        new_chain = [None for i in range(len(chain))]
        for i in range(len(chain)):
            new_chain[id_map[i]] = chain[i]
        new_chains.append(new_chain)
    new_connected_pairs = []
    for i, j, t in connected_pairs:
        new_i = id_map[i]
        new_j = id_map[j]
        new_connected_pairs.append((new_i, new_j, t))
    new_connected_pairs = sorted(new_connected_pairs, key=lambda x:min([x[0], x[1]]))
    return new_aas, new_aa_units, new_chains, new_connected_pairs

def aa_xyz(conformer, aa_idx):
    # 获取指定原子的二维坐标
    coords = [conformer.GetAtomPosition(idx) for idx in aa_idx]
    x_mean = sum(coord.x for coord in coords) / len(coords)
    y_mean = sum(coord.y for coord in coords) / len(coords)
    return (x_mean, y_mean)

def sequence_map(m, aa_units, chain, connected_pairs, isdisplay=True):
    aa_locations = []
    m.Compute2DCoords() # 确保分子具有二维坐标
    conformer = m.GetConformer()
    for i in aa_units:
        aa_locations.append(aa_xyz(conformer, i))
    ## plot
    plt.figure(figsize=(6, 6))
    x = [i[0] for i in aa_locations]
    y = [i[1] for i in aa_locations]
    plt.scatter(x, y, color='gray', marker='o', s=300)
    # 添加数字标记
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i+1), color='white', fontsize=10, ha='center', va='center')
    for i, j, t in connected_pairs:
        pairx = [aa_locations[i][0], aa_locations[j][0]]
        pairy = [aa_locations[i][1], aa_locations[j][1]]
        ls = '--' if t == 'side chain' else '-'
        plt.plot(pairx, pairy, color='gray', ls=ls)
    cmap = cm.get_cmap('tab20', len(chain))
    shift = (max(x)-min(x))*0.02
    for aa_i in range(len(aa_units)):
        aa_symbol = chain[aa_i][0]
        #print(chain[aa_i])
        #aa_symbol = aa_symbol + '*' if chain[aa_i][1]!=chain[aa_i][-1] else aa_symbol
        plt.annotate(aa_symbol, np.array(aa_locations[aa_i]) + shift,fontsize=18, color=cmap(aa_i), zorder=999)
    # 去除坐标轴
    plt.axis('off')
    
    # 保存为SVG到内存中的字符串
    f = BytesIO()
    plt.savefig(f, format="svg")
    plt.close()
    
    # 获取SVG图像的字符串
    f.seek(0)
    svg_data = f.read().decode('utf-8')
    # 清除命名空间前缀
    svg_data = svg_data.replace('xmlns="http://www.w3.org/2000/svg" ', '')
    if isdisplay:
        display(SVG(svg_data))
    return svg_data
    

def sequence_str(chain, connected_pairs):
    seq = [i[0] for i in chain]
    link_idx = 1
    for i, j, t in connected_pairs:
        if t=='side chain' or abs(i-j)!=1:
            seq[i] = seq[i]+'('+str(link_idx)+')'
            seq[j] = seq[j]+'('+str(link_idx)+')'
            link_idx += 1
    return '--'.join(seq)
    
def transform(smiles, path='monomer.tsv'):
    report = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <title>Structure 2 Sequence Report</title>
        <style>
          .aacontainer {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
          }

          .aabox {
            width: 20%; /* 每个box占据大约30%的宽度 */
            margin: 10px;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.2); /* 阴影效果 */
            text-align: center; /* 内部元素居中 */
            flex-direction: column; /* 子元素纵向排列 */
            justify-content: center; /* 子元素在主轴方向上居中 */
            align-items: center; /* 子元素在交叉轴方向上居中 */
          }

          .aabox p {
            margin: 10px 0;
            text-align: center; /* 文本居中 */
          }

          .aabox svg {
            width: 100%;
            height: auto;
          }
          
          .resbox {
            width: 45%; 
            margin: 10px;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.2); /* 阴影效果 */
            text-align: center; /* 内部元素居中 */
            flex-direction: column; /* 子元素纵向排列 */
            justify-content: center; /* 子元素在主轴方向上居中 */
            align-items: center; /* 子元素在交叉轴方向上居中 */
          }
          .resbox p {
            margin: 10px 0;
            text-align: center; /* 文本居中 */
          }

          .resbox svg {
            width: 100%;
            height: auto;
          }
        </style>
        </head>
        <body>
        <h1>Structure 2 Sequence Report</h1>
        <p>Structure-to-Sequence (s2s) is a computing process based on <a href='http://www.rdkit.org/'>RDkit</a> and the characteristics of cyclic peptide sequences, which can convert cyclic peptide SMILES into sequence information. This process mainly relies on the completeness of the <a href='https://www.biosino.org/iMAC/cyclicpepedia/stru2seq'>monomer reference library</a>. You can access our default monomer reference library through <a href='https://www.biosino.org/iMAC/cyclicpepedia/download'>download link</a>. The details of s2s are available on <a href='https://github.com/dfwlab/cyclicpepedia'>dfwlab/cyclicpepedia</a> on Github. And you can use this tool online on the <a href='https://www.biosino.org/iMAC/cyclicpepedia/stru2seq'>cyclicpepedia</a>.</p>
        <br/>
        <p><b>Version</b> : 1.0.1 (2023-12-26)</p>
        <hr/>
        """
    report_footer = '''
        </body>
        </html>
        '''
    # 读取结构
    sub_report = '''<h3>Load SMILES : </h3><p>SMILES : {smiles}</p><p>{loadstate}</p><hr/>'''
    try:
        m = Chem.MolFromSmiles(smiles)
        sub_report = sub_report.format(smiles=smiles, loadstate = 'SMILES is corrected!')
        report += sub_report
    except:
        sub_report = sub_report.format(smiles=smiles, loadstate = 'Load SMILES error! Check you input!')
        return report + sub_report + report_footer
    
    # 确定骨架，重编号骨架
    sub_report = '''<h3>Identify peptide skeleton and renumber atoms</h3><div>{backbone}</div><hr/>'''
    is_backbone = False
    try:
        backbone_idx = detect_backbone(m)
        m, backbone_idx = order_backbone(m, backbone_idx)
        svg = highlight_atom(m, [backbone_idx], isdisplay=False)
        sub_report = sub_report.format(backbone = svg)
        report += sub_report
        is_backbone = True
    except:
        sub_report = sub_report.format(backbone = 'No single main skeleton found in the peptide. Custom amino acid sorting strategy will be used!')
        report += sub_report
        is_backbone = False
        #return report + sub_report + report_footer
    
    # 识别氨基酸单元
    sub_report = '''<h3>Identify amino acid units</h3><div>{units}</div><hr/>'''
    try:
        aa_units = split_aa_unit(m)
        svg = highlight_atom(m, aa_units, transparency=0.5, isdisplay=False) # 重叠部分以固定透明度混合颜色绘制
        sub_report = sub_report.format(units = svg)
        report += sub_report
    except:
        sub_report = sub_report.format(units = 'Can not find amino acid unit in peptide!')
        return report + sub_report + report_footer
    
    # 获取完整氨基酸结构(切分肽键)
    aas_sub_report = '''<h3>Obtain complete amino acid structures</h3><div>{aas}</div><hr/>'''
    try:
        aas, atom_mappings = get_complete_aas(m, aa_units)
        ################ 暂不输出 aas 结果 ################
        #report += sub_report
    except:
        aas_sub_report = aas_sub_report.format(aas = 'Error!')
        return report + aas_sub_report + report_footer
    
    # 导入氨基酸单元库，识别氨基酸种类，按分子结构组装成氨基酸链
    sub_report = '''<h3>Identify amino acids based on the monomer reference library</h3>'''
    #try:
    ref_essentials, ref_others = reference_aa_monomer(path)
    connected_pairs = get_connected_pairs(m, atom_mappings)
    chains = aa_matching(aas, atom_mappings, connected_pairs, ref_essentials, ref_others)
    sub_report += '''<p>Number of chain(s) identified from the structure: <b>{nchains}</b></p>'''.format(nchains=len(chains))
    if not is_backbone:
        order_chain = get_connected_chain(connected_pairs, len(aas))
        aas, aa_units, chains, connected_pairs = reorder_result(order_chain, aas, aa_units, chains, connected_pairs)

    ######## 重新生成 aas 结果 （按新的编号） ########
    temp = '<div class="aacontainer">'
    i = 1
    for aa in aas:
        svg = highlight_atom(aa, [[]], isdisplay=False)
        temp += '<div class="aabox"><p>Amino acid '+str(i)+'</p>'+svg+'</div>'
        i += 1
    temp += '</div>'
    aas_sub_report = aas_sub_report.format(aas = temp)
    report += aas_sub_report
    ##############################################
    ci = 1
    for chain in chains:
        sub_report += '''<h4> > Chain {ci} :</h4>'''.format(ci=ci)
        sub_report += '''<p><b>Amino acid sequence :</b> {seq}</p>'''.format(seq=sequence_str(chain, connected_pairs))
        sub_report += '<div class="aacontainer">'
        sub_report += '''<div class="resbox"><p><b>Amino acid mapping</b></p>{svg}</div>'''.format(svg=highlight_atom(m, [i[5] for i in chain], isdisplay=False))
        sub_report += '''<div class="resbox"><p><b>Amino acid location</b></p>{svg}</div>'''.format(svg=sequence_map(m, aa_units, chain, connected_pairs, isdisplay=False))
        sub_report += '</div>'

        ######## 生成 mapping 结果 ########
        aas_map_sub_report = '''<p><b>Matched amino acid from monomer reference library</b></p><div>{aas}</div><hr/>'''
        refs = {**ref_essentials, **ref_others}
        temp = '<div class="aacontainer">'
        i = 1
        for aa in chain:
            ref_aa = refs[aa[0]]
            #print(aa, ref_aa)
            svg = highlight_atom(ref_aa[0], [[]], isdisplay=False)
            temp += '<div class="aabox"><p>Amino acid '+str(i)+': '+aa[0]+'</p>'+svg+'</div>'
            i += 1
        temp += '</div>'
        aas_map_sub_report = aas_map_sub_report.format(aas = temp)
        sub_report += aas_map_sub_report
        ##############################################
        ci += 1
    report += sub_report
    #except:
    #    sub_report += '''<p>Error!</p>'''
    #    return report + sub_report + report_footer
    
    return report + report_footer
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

