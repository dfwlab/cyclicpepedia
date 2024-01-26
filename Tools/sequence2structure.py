#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: sequence2structure.py
Author: Dingfeng Wu
Creator: Dingfeng Wu
Date Created: 2024-01-03
Last Modified: 2023-01-05
Version: 1.0.0
License: MIT License
Description: Sequence-to-Structure (Seq2struc) is a computing process based on RDkit and the characteristics of cyclic peptide sequences, which can create cyclic peptide sequecne and convert sequence to cyclic peptide SMILES.

Copyright Information: Copyright (c) 2023 dfwlab (https://dfwlab.github.io/)

The code in this script can be used under the MIT License.
"""

import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG

def plot_molecule(mol, w=600, h=600, isdisplay=False):
    # 设置绘图选项，调整分子的大小
    d2d = rdMolDraw2D.MolDraw2DSVG(w, h)
    
    # 绘制分子
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    # 显示图像
    svg = d2d.GetDrawingText()
    if isdisplay:
        display(SVG(svg.replace('svg:', '')))
    else:
        return svg

def plot_smiles(smiles, w=600, h=600, isdisplay=False):
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
    if isdisplay:
        display(SVG(svg.replace('svg:', '')))
    else:
        return svg
    
def code2symbol(code):
    AminoAcids = [# 20个必须氨基酸
              ('A', 'ALA', 'C[C@H](N)C=O'), ('C', 'CYS', 'N[C@H](C=O)CS'), ('D', 'ASP', 'N[C@H](C=O)CC(=O)O'), 
              ('E', 'GLU', 'N[C@H](C=O)CCC(=O)O'), ('F', 'PHE', 'N[C@H](C=O)Cc1ccccc1'), ('G', 'GLY', 'NCC=O'), 
              ('H', 'HIS', 'N[C@H](C=O)Cc1c[nH]cn1'), ('I', 'ILE', 'CC[C@H](C)[C@H](N)C=O'), 
              ('K', 'LYS', 'NCCCC[C@H](N)C=O'), ('L', 'LEU', 'CC(C)C[C@H](N)C=O'), ('M', 'MET', 'CSCC[C@H](N)C=O'), 
              ('N', 'ASN', 'NC(=O)C[C@H](N)C=O'), ('P', 'PRO', 'O=C[C@@H]1CCCN1'), 
              ('Q', 'GLN', 'NC(=O)CC[C@H](N)C=O'), ('R', 'ARG', 'N=C(N)NCCC[C@H](N)C=O'), 
              ('S', 'SER', 'N[C@H](C=O)CO'), ('T', 'THR', 'C[C@@H](O)[C@H](N)C=O'), ('V', 'VAL', 'CC(C)[C@H](N)C=O'), 
              ('W', 'TRP', 'N[C@H](C=O)Cc1c[nH]c2ccccc12'), ('Y', 'TYR', 'N[C@H](C=O)Cc1ccc(O)cc1'), ]
    c2s = {i[1]:[i[0], i[2]] for i in AminoAcids}
    if code.upper() in c2s.keys():
        return c2s[code.upper()]
    else:
        return None

def detect_backbone(m):
    bbsmiles = "C(=O)CN"*len(m.GetSubstructMatches(Chem.MolFromSmiles("NCC=O"))) # generate backbone SMILES
    backbone = m.GetSubstructMatches(Chem.MolFromSmiles(bbsmiles))[0]
    return backbone

def link_aa_by_peptide_bond(mol, c_index, n_index):
    o_index = None
    h_index = None
    # 查找与该碳原子相连的羟基的氧和氢原子
    for atom in mol.GetAtomWithIdx(c_index).GetNeighbors():
        if atom.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(atom.GetIdx(), c_index).GetBondType() == Chem.BondType.SINGLE:
            o_index = atom.GetIdx()
            for h_atom in atom.GetNeighbors():
                if h_atom.GetAtomicNum() == 1:  # 氢原子
                    h_index = h_atom.GetIdx()
    #print(o_index, h_index)
    # 创建一个可编辑的分子
    emol = Chem.EditableMol(mol)
    # 注意：您需要根据实际的氨基酸残基位置调整原子索引
    emol.RemoveAtom(h_index) if h_index else 1
    emol.RemoveAtom(o_index)
    emol.AddBond(c_index, n_index, order=Chem.rdchem.BondType.SINGLE)
    # 获取修改后的分子
    mol = emol.GetMol()
    return mol

def create_peptide_of_essentialAA(sequence, cyclic=True):
    try:
        mol = Chem.MolFromSequence(sequence)
    except:
        return None
    if cyclic:
        backbone = detect_backbone(mol)
        c_index = backbone[0]
        n_index = backbone[-1]
        mol = link_aa_by_peptide_bond(mol, c_index, n_index)
    return mol

def seq2stru_essentialAA(sequence, cyclic=True):
    try:
        if '-' in sequence:
            sequence = [code2symbol(code)[0] for code in sequence.split('-')]
            sequence = ''.join(sequence)
        peptide = create_peptide_of_essentialAA(sequence, cyclic=cyclic)
        peptide = Chem.RemoveHs(peptide) # 移除所有隐式氢原子
        Chem.AssignAtomChiralTagsFromStructure(peptide) # 重新计算所有隐式氢原子
        smiles = Chem.MolToSmiles(peptide, canonical=True)
        return smiles, peptide
    except:
        return None, None

def load_nomomer(path):
    monomers = pd.read_csv(path, sep='\t', index_col=0)
    return monomers
    
def reference_aa_monomer(monomers):
    references = {}
    for i in monomers.index:
        smile = monomers.loc[i, 'Smiles']
        code = monomers.loc[i, 'Code']
        weight = float(monomers.loc[i, 'Weight'])
        symbol = monomers.loc[i, 'Symbol']
        symbol = code if str(symbol).strip()=='' or str(symbol).upper().strip()=='NAN' else symbol
        try:
            aa = Chem.MolFromSmiles(smile)
            match = aa.GetSubstructMatches(Chem.MolFromSmiles("NCC=O"))
            if match and match[0]:
                num_atoms = aa.GetNumAtoms()
                references[code] = [aa, symbol, num_atoms, match[0]]
        except:
            pass
    return references

def connect_two_aa_with_peptide_bond(aa1, c_index1, aa2, c_index2, n_index2):
    c_index2, n_index2 = c_index2 + aa1.GetNumAtoms() - 1, n_index2 + aa1.GetNumAtoms() - 1
    peptide = Chem.CombineMols(aa1, aa2)
    peptide = link_aa_by_peptide_bond(peptide, c_index1, n_index2)
    return peptide, c_index2

def create_peptide_of_no_essentialAA(sequence, references, cyclic=True):
    peptide, pep_idx = references[sequence[0]][0], references[sequence[0]][-1]
    pep_c_index, first_n_index = pep_idx[2], pep_idx[0]
    for aa_code in sequence[1:]:
        aa, idx = references[aa_code][0], references[aa_code][-1]
        c_index, n_index = idx[2], idx[0]
        peptide, pep_c_index = connect_two_aa_with_peptide_bond(peptide, pep_c_index, aa, c_index, n_index)
    if cyclic:
        peptide = link_aa_by_peptide_bond(peptide, pep_c_index, first_n_index)
    return peptide

def seq2stru_no_essentialAA(sequence, references, cyclic=True):
    try:
        sequence = [i.strip() for i in sequence.split('--')]
        peptide = create_peptide_of_no_essentialAA(sequence, references, cyclic=cyclic)
        smiles = Chem.MolToSmiles(peptide, canonical=True)
        peptide = Chem.RemoveHs(peptide) # 移除所有隐式氢原子
        Chem.AssignAtomChiralTagsFromStructure(peptide) # 重新计算所有隐式氢原子
        return smiles, peptide
    except:
        return None, None

def get_molblock(mol):
    return Chem.MolToMolBlock(mol, forceV3000=True)

def get_molblock_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToMolBlock(mol, forceV3000=True)






