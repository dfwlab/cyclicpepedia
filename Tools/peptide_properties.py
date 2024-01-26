#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: peptide_properties.py
Author: Dingfeng Wu
Creator: Dingfeng Wu
Date Created: 2023-12-30
Last Modified: 2024-01-13
Version: 1.0.1
License: MIT License
Description: The peptide properties (PP) algorithm is developed by CyclicPepedia based on RDkit and Peptides to predict structure and sequence properties.

Copyright Information: Copyright (c) 2023 dfwlab (https://dfwlab.github.io/)

The code in this script can be used under the MIT License.
"""

import re
import pandas as pd
from IPython.display import SVG

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Crippen, MolSurf, rdMolDescriptors, Lipinski, rdFingerprintGenerator, MACCSkeys, RDKFingerprint
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)  # 只显示 CRITICAL 级别的日志

import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.pandas2ri import py2rpy, rpy2py
pandas2ri.activate()
ro.r.options(warn=-1) # 关闭R的warning输出

# https://www.cryst.bbk.ac.uk/education/AminoAcid/the_twenty.html
# https://www.sciencefriday.com/wp-content/uploads/2018/07/amino-acid-abbreviation-chart.pdf
AminoAcids = [('A', 'ALA', 'C[C@H](N)C=O', 'Alanine'), 
              ('C', 'CYS', 'N[C@H](C=O)CS', 'Cysteine'), 
              ('D', 'ASP', 'N[C@H](C=O)CC(=O)O', 'Aspartic acid'), 
              ('E', 'GLU', 'N[C@H](C=O)CCC(=O)O', 'Glutamic acid'), 
              ('F', 'PHE', 'N[C@H](C=O)Cc1ccccc1', 'Phenylalanine'), 
              ('G', 'GLY', 'NCC=O', 'Glycine'), 
              ('H', 'HIS', 'N[C@H](C=O)Cc1c[nH]cn1', 'Histidine'), 
              ('I', 'ILE', 'CC[C@H](C)[C@H](N)C=O', 'Isoleucine'), 
              ('K', 'LYS', 'NCCCC[C@H](N)C=O', 'Lysine'), 
              ('L', 'LEU', 'CC(C)C[C@H](N)C=O', 'Leucine'), 
              ('M', 'MET', 'CSCC[C@H](N)C=O', 'Methionine'), 
              ('N', 'ASN', 'NC(=O)C[C@H](N)C=O', 'Asparagine'), 
              ('P', 'PRO', 'O=C[C@@H]1CCCN1', 'Proline'), 
              ('Q', 'GLN', 'NC(=O)CC[C@H](N)C=O', 'Glutamine'), 
              ('R', 'ARG', 'N=C(N)NCCC[C@H](N)C=O', 'Arginine'), 
              ('S', 'SER', 'N[C@H](C=O)CO', 'Serine'), 
              ('T', 'THR', 'C[C@@H](O)[C@H](N)C=O', 'Threonine'), 
              ('V', 'VAL', 'CC(C)[C@H](N)C=O', 'Valine'), 
              ('W', 'TRP', 'N[C@H](C=O)Cc1c[nH]c2ccccc12', 'Tryptophan'), 
              ('Y', 'TYR', 'N[C@H](C=O)Cc1ccc(O)cc1', 'Tyrosine'), 
              ('O', 'PYL', '', 'Pyrrolysine'), # 非必需
              ('U', 'SEC', '', 'Selenocysteine'), # 非必需
              ('B', 'ASX', '', 'Aspartic acid or Asparagine'), # 必需组合
              ('Z', 'GLX', '', 'Glutamic acid or Glutamine'), # 必需组合
             ]

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
################## 化学性质 ##################
def cal_chemial_physical_properties(molecule):
    # 理化性质函数
    properties = {
        'Exact_Mass': Descriptors.ExactMolWt(molecule),
        'Topological_Polar_Surface_Area': CalcTPSA(molecule),
        'Complexity': Descriptors.FpDensityMorgan1(molecule),
        'Crippen_LogP': Descriptors.MolLogP(molecule), # 计算LogP Crippen
        'Heavy_Atom_Count': Lipinski.HeavyAtomCount(molecule),
        'Hydrogen_Bond_Donor_Count': Lipinski.NumHDonors(molecule),
        'Hydrogen_Bond_Acceptor_Count': Lipinski.NumHAcceptors(molecule),
        'Rotatable_Bond_Count': Lipinski.NumRotatableBonds(molecule),
        'Formal_Charge': Chem.GetFormalCharge(molecule),
        'Refractivity': Descriptors.MolMR(molecule),
        'Number_of_Rings': rdMolDescriptors.CalcNumRings(molecule),
        'Number_of_Atoms': molecule.GetNumAtoms(),
    }
    return properties

def cal_rules(properties):
    # Lipinski 规则五
    lipinski_rule_of_five = properties['Hydrogen_Bond_Donor_Count'] <= 5 and \
                            properties['Hydrogen_Bond_Acceptor_Count'] <= 10 and \
                            properties['Exact_Mass'] <= 500 and \
                            properties['Crippen_LogP'] <= 5
    
    # Veber 规则
    vebers_rule = properties['Topological_Polar_Surface_Area'] <= 140 and \
                  properties['Rotatable_Bond_Count'] <= 10
    
    # 检查 Ghose Filter 条件
    ghose_filter = 160 <= properties['Exact_Mass'] <= 480 and \
                   0.4 <= properties['Crippen_LogP'] <= 5.6 and \
                   20 <= properties['Number_of_Atoms'] <= 70
    
    properties['Rule_of_Five'] = lipinski_rule_of_five
    properties["Veber_Rule"] = vebers_rule
    properties['Ghose_Filter'] = ghose_filter
    return properties

################## 分子指纹 ##################
def cal_RDKit_fingerprint(mol):
    # 生成RDKit指纹
    fp = RDKFingerprint(mol)
    bit_vector = fp.ToBitString()
    return bit_vector
    
def cal_daylight_like_fingerprint(mol):
    # 生成拓扑指纹
    fp = rdFingerprintGenerator.GetFPs([mol])
    bit_vector = fp[0].ToBitString()
    return bit_vector

def cal_morgan_fingerprint(mol):
    # 生成Morgan指纹（半径为2）
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    bit_vector = morgan_fp.ToBitString()
    return bit_vector

def cal_MACCS_keys(mol):
    # 生成 MACCS keys 指纹
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    bit_vector = maccs_fp.ToBitString()
    return bit_vector

def chemial_physical_properties_from_smiles(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except:
        return {}
    properties = cal_chemial_physical_properties(molecule)
    properties = cal_rules(properties)
    properties['RDKit_Fingerprint'] = cal_RDKit_fingerprint(molecule)
    properties['Daylight_like_Fingerprint'] = cal_daylight_like_fingerprint(molecule)
    properties['Morgan_Fingerprint'] = cal_morgan_fingerprint(molecule)
    properties['MACCS_Keys'] = cal_MACCS_keys(molecule)
    return properties

################## 序列性质 ##################
def cal_aa_composition():
    res = r("aaComp(sequence)")[0]
    msg = pd.DataFrame([['Tiny', '(A+C+G+S+T) '], ['Small', '(A+B+C+D+G+N+P+S+T+V)'], ['Aliphatic', '(A+I+L+V)'], 
          ['Aromatic', '(F+H+W+Y)'], ['Non-polar', '(A+C+F+G+I+L+M+P+V+W+Y)'], 
          ['Polar', '(D+E+H+K+N+Q+R+S+T+Z)'], ['Charged', '(B+D+E+H+K+R+Z)'], 
          ['Basic', '(H+K+R)'], ['Acidic', '(B+D+E+Z)']], columns=['Property', 'Residues'], index=range(9))
    res = pd.concat([msg, pd.DataFrame(res, columns=['Number', 'Mole%'], index=range(9))], axis=1, sort=False)
    return res

def cal_hydrophobicity():
    scales = ["Aboderin", "AbrahamLeo", "Argos", "BlackMould", "BullBreese", "Casari", "Chothia", "Cid", 
          "Cowan3.4", "Cowan7.5", "Eisenberg", "Engelman", "Fasman", "Fauchere", "Goldsack", "Guy", 
          "HoppWoods", "Janin", "Jones", "Juretic", "Kidera", "Kuhn", "KyteDoolittle", "Levitt", "Manavalan", 
          "Miyazawa", "Parker", "Ponnuswamy", "Prabhakaran", "Rao", "Rose", "Roseman", "Sweet", "Tanford", 
          "Welling", "Wilson", "Wolfenden", "Zimmerman", "interfaceScale_pH8", "interfaceScale_pH2", 
          "octanolScale_pH8", "octanolScale_pH2", "oiScale_pH8", "oiScale_pH2"]
    hydrophobicities = []
    for scale in scales:
        hydrophobicities.append(r("hydrophobicity(sequence, scale = '"+scale+"')")[0])
    hydrophobicities = pd.DataFrame(hydrophobicities, index=scales, columns=['Indices'])
    return hydrophobicities

def cal_pI():
    pKscales = ["Bjellqvist", "EMBOSS", "Murray", "Sillero", "Solomon", "Stryer", "Lehninger", "Dawson", "Rodwell"]
    pIs = []
    for scale in pKscales:
        pIs.append(r("pI(sequence, pKscale = '"+scale+"')")[0])
    pIs = pd.DataFrame(pIs, index=pKscales, columns=['Indices'])
    return pIs

def predict_properties_from_sequence(sequence):
    importr('Peptides')
    _ = r.assign('sequence', sequence)
    # 1. composition
    try:
        aa_composition = cal_aa_composition()
    except:
        aa_composition = pd.DataFrame()
    # 2. aliphatic_index
    try:
        aliphatic_index = r("aIndex(sequence)")[0]
    except:
        aliphatic_index = ''
    # 3. blosum62
    try:
        blosum62 = r("blosumIndices(sequence)")[0]
        blosum62 = pd.DataFrame(blosum62, index=['BLOSUM'+str(i) for i in range(1, 11)], columns=['Indices'])
    except:
        blosum62 = pd.DataFrame()
    # 4. boman index
    try:
        boman_index = r("boman(sequence)")[0]
    except:
        boman_index = ''
    # 5. charge
    try:
        charge = r("charge(sequence)")[0]
    except:
        charge = ''
    # 6. crucianiProperties
    try:
        res = r("crucianiProperties(sequence)")[0]
        cruciani = pd.DataFrame(res, index=['PP1', 'PP2', 'PP3'], columns=['Indices'])
        cruciani['Metric'] = ['Polarity', 'Hydrophobicity', 'H-bonding']
    except:
        cruciani = pd.DataFrame()
    # 7. fasgaiVectors
    try:
        res = r("fasgaiVectors(sequence)")[0]
        fasgai = pd.DataFrame(res, index=['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], columns=['Indices'])
        fasgai['Metric'] = ['Hydrophobicity index', 'Alpha and turn propensities', 'Bulky properties', 
                            'Compositional characteristic index', 'Local flexibility', 'Electronic properties']
    except:
        fasgai = pd.DataFrame()
    # 8. hydrophobic moment
    try:
        hmoment_a = r("hmoment(sequence, angle = 100, window = 11)")[0] # a-helix
        hmoment_b = r("hmoment(sequence, angle = 160, window = 11)")[0] # b-sheet
        hmoment = pd.DataFrame([hmoment_a, hmoment_b], index=['a-helix', 'b-sheet'], columns=['Indices'])
        hmoment['angle'] = [100, 160]
        hmoment['window'] = [11, 11]
    except:
        hmoment = pd.DataFrame()
    # 9. hydrophobicity
    try:
        hydrophobicity = cal_hydrophobicity()
    except:
        hydrophobicity = pd.DataFrame()
    # 10. instability index
    try:
        instability = r("instaIndex(sequence)")[0]
    except:
        instability = ''
    # 11. kidera Factors
    try:
        kidera = r("kideraFactors(sequence)")[0]
        kidera = pd.DataFrame(kidera, index=['KF'+str(i) for i in range(1, 11)], columns=['Indices'])
        kidera['Metric'] = ['Helix/bend preference', 'Side-chain size', 'Extended structure preference', 
                            'Hydrophobicity', 'Double-bend preference', 'Partial specific volume', 
                            'Flat extended preference', 'Occurrence in alpha region', 'pK-C', 'Surrounding hydrophobicity']
    except:
        kidera = pd.DataFrame()
    # 12. amino acid length
    try:
        lengthpep = r("lengthpep(sequence)")[0]
    except:
        lengthpep = ''
    # 13. membpos
    try:
        membpos1 = pd.DataFrame(r("membpos(sequence, angle = 100)")[0], index=['Pep', 'H', 'uH', 'MembPos']).T
        membpos1['angle'] = 100
        membpos2 = pd.DataFrame(r("membpos(sequence, angle = 160)")[0], index=['Pep', 'H', 'uH', 'MembPos']).T
        membpos2['angle'] = 160
        membpos = pd.concat([membpos1, membpos2], axis=0, sort=False)
        membpos.index = range(membpos.shape[0])
    except:
        membpos = pd.DataFrame()
    # 14. mswhimScores
    try:
        mswhimScores = pd.DataFrame(r("mswhimScores(sequence)")[0], 
                                    index=['MSWHIM1', 'MSWHIM2', 'MSWHIM3'], columns=['Indices'])
    except:
        mswhimScores = pd.DataFrame()
    # 15. isoelectic point (pI)
    try:
        pIs = cal_pI()
    except:
        pIs = pd.DataFrame()
    # 16. protFP
    try:
        protFP = pd.DataFrame(r("protFP(sequence)")[0], index=['ProtFP'+str(i) for i in range(1, 9)], columns=['Indices'])
    except:
        protFP = pd.DataFrame()
    # 17. stScales
    try:
        stScales = pd.DataFrame(r("stScales(sequence)")[0], index=['ST'+str(i) for i in range(1, 9)], columns=['Indices'])
    except:
        stScales = pd.DataFrame()
    # 18. tScales
    try:
        tScales = pd.DataFrame(r("tScales(sequence)")[0], index=['T'+str(i) for i in range(1, 6)], columns=['Indices'])
    except:
        tScales = pd.DataFrame()
    # 19. vhseScales
    try:
        vhseScales = pd.DataFrame(r("vhseScales(sequence)")[0], 
                                  index=['VHSE'+str(i) for i in range(1, 9)], columns=['Indices'])
    except:
        vhseScales = pd.DataFrame()
    # 20. zScales
    try:
        zScales = pd.DataFrame(r("zScales(sequence)")[0], index=['Z'+str(i) for i in range(1, 6)], columns=['Indices'])
    except:
        zScales = pd.DataFrame()
    return {'aa_comp':aa_composition, 'aliphatic_index':aliphatic_index, 'blosum62':blosum62, 
            'boman_index':boman_index, 'charge':charge, 'cruciani':cruciani, 'fasgai':fasgai, 
            'hmoment':hmoment, 'hydrophobicity':hydrophobicity, 'instability':instability, 'kidera':kidera, 
            'lengthpep':lengthpep, 'membpos':membpos, 'mswhimScores':mswhimScores, 'pIs':pIs, 
            'protFP':protFP, 'stScales':stScales, 'tScales':tScales, 'vhseScales':vhseScales, 'zScales':zScales}

def predict_basic_properties_from_sequence(sequence):
    importr('Peptides')
    _ = r.assign('sequence', sequence)
    try:
        boman_index = r("boman(sequence)")[0]
    except:
        boman_index = ''
    # 5. charge
    try:
        charge = r("charge(sequence)")[0]
    except:
        charge = ''
    # 2. aliphatic_index
    try:
        aliphatic_index = r("aIndex(sequence)")[0]
    except:
        aliphatic_index = ''
    # 10. instability index
    try:
        instability = r("instaIndex(sequence)")[0]
    except:
        instability = ''
    try:
        aa_composition = calculate_amino_acid_composition(sequence)
    except:
        aa_composition = ''
    return [boman_index, charge, aliphatic_index, instability, aa_composition]

def calculate_amino_acid_composition(sequence):
    aa_list = [i[0] for i in AminoAcids]
    counts = [(aa, sequence.count(aa) if aa in sequence else 0) for aa in aa_list]
    return counts
    
################## 结构格式互转 ##################
# 自动识别并读人SMILES InChI Molblock SDFblock PDBblock
# https://chemistry.stackexchange.com/questions/34563/pubchem-inchi-smiles-and-uniqueness
# 以InChi=1S/…开头的InChi标识符是标准InChI。在标准InChI中，InChI标识符“对于任何移动氢原子的排列都必须相同”
# 以InChI=1/…开始为非标准InChI，包括一个以/f开头的额外层（fixed-hydrogen layer）。
# 标准InChI生成结构与SMILES不同，而非标准InChI生成结构与SMILES相同，非标准InChI生成生成代码：Chem.MolToInchi(mol1, options='/FixedH')
# 自动识别并读人SMILES InChI Molblock SDFblock PDBblock
def load_molecule(input_mol):
    if 'InChI' in input_mol:
        method = Chem.MolFromInchi
        mol_type = 'INCHI'
    elif '\n' in input_mol:
        if 'ATOM' in input_mol.upper() or 'HETATM' in input_mol.upper() or 'CONECT' in input_mol.upper():
            method = Chem.MolFromPDBBlock
            mol_type = 'PDB'
        else:
            method = Chem.MolFromMolBlock
            mol_type = 'MOL'
    else:
        method = Chem.MolFromSmiles
        mol_type = 'SMILES'
    mol = method(input_mol)
    if mol_type=='PDB':
        mol.RemoveAllConformers() # 去除PDB的3D构象，以免绘图异常
    return mol_type, mol

# 生成 SMILES InChI InChIKey Molblock SDFblock PDBblock
def output_molecule(mol, pdbblock=None, conformation=None):
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    inchi = Chem.MolToInchi(mol, options='/FixedH')
    inchikey = Chem.InchiToInchiKey(inchi)
    molblock = Chem.MolToMolBlock(mol, includeStereo=True)
    ### PDBblock需要通过空间结构保存手性, 因此其他格式转PDB时需要先创建3D构象;
    if pdbblock is None:
        pdbblock = Chem.MolToPDBBlock(conformation)
    return {'smiles':smiles, 'inchi':inchi, 'inchikey':inchikey, 'molblock':molblock, 'pdbblock':pdbblock}

def predict_3d_conformation(mol):
    #molecule = Chem.MolFromSmiles(smiles)
    mol_3d = Chem.AddHs(mol.__copy__())  # 添加氢
    AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())  # 生成 3D 坐标
    return mol_3d

def mol_optimize(mol):
    # 能量最小化：UFF (Universal Force Field)适用于小分子的力场；多肽建模和优化，专门的生物分子建模工具（如 AMBER、CHARMM 或 GROMACS）会更合适
    AllChem.UFFOptimizeMolecule(mol)  # 使用 UFF 力场进行优化
    return mol

def mol2molblock(mol):
    return Chem.MolToMolBlock(mol)

def mol2pdbblock(mol):
    return Chem.MolToPDBBlock(mol)

################## 序列格式互转 ##################
def replace_hyphen(match):
    # 替换括号内的所有 hyphen (-) 为 tilde (~)
    return match.group(0).replace('-', '~~')
    
def read_iupac_condensed(sequence, sep='-'):
    # 0. 确认是否有盐或多条链
    if '.' in sequence:
        sequence = sequence.split('.')[0].strip()
    # 1. 确认是否有cyclo[]成环信息
    iscyclo = True if 'cyclo' in sequence else False
    sequence = sequence.replace('cyclo', '').strip('[]')# 移除 'cyclo' 和方括号
    sequence = re.sub(r'\([^)]*\)', replace_hyphen, sequence)# 替换（）中的-为～～，（）中为修饰
    items = sequence.split(sep)
    # 2. 确认首尾修饰信息
    header = ''
    tail = ''
    if items[0] in ['H', 'NH2', 'Unk']:
        header = items[0]
        items = items[1:]
    if items[-1] in ['H', 'NH2', 'Unk']:
        tail = items[-1]
        items = items[:-1]
    # 3. 获取氨基酸，DL-修饰添加进氨基酸，首位修饰添加进氨基酸
    amino_acids = []
    edge_marks = []
    pred = ''
    max_edge_mark = 0
    for item in items:
        if item in ['DL', 'D', 'L']:
            pred = item+'-'
            continue
        amino_acid = (pred+item).replace('~~', '-').strip()
        edge_mark = []
        while True:
            edge_mark_match = re.search(r'\(\d+\)$', amino_acid)
            if edge_mark_match:
                amino_acid = amino_acid.replace(edge_mark_match[0], '').strip()
                em = int(edge_mark_match[0][1:-1])
                max_edge_mark = max([max_edge_mark, em])
                edge_mark.append(em)
            else:
                break
        amino_acids.append(amino_acid)
        edge_marks.append(edge_mark)
        pred = ''
    if header:
        amino_acids[0] = header+'-'+amino_acids[0]
    if tail:
        amino_acids[-1] = amino_acids[-1]+'-'+tail
    # 4. 处理最后空格但有edge标签的
    if amino_acids[-1]=='' and edge_marks:
        amino_acids = amino_acids[:-1]
        edge_mark = edge_marks[-1][:]
        edge_marks = edge_marks[:-1]
        edge_marks[-1].extend(edge_mark)
    # 5. 处理氨基酸N(1)标注
    for i, aa in enumerate(amino_acids):
        match = re.search(r'N\((\d)\)', aa)
        if match:
            amino_acids[i] = re.sub(r'N\((\d)\)', 'N-', aa)
            edge_marks[i].append(int(match.group(1)))
        
    #print(amino_acids)
    #print(edge_marks)
    # 6. edges信息处理
    edges = []
    for i in range(1, max_edge_mark+1):
        pair = []
        for idx in range(len(edge_marks)):
            if i in edge_marks[idx]:
                pair.append(idx)
        edges.append((pair[0], pair[1]))
    if iscyclo:
        edges.append((0, len(amino_acids)-1))
    return amino_acids, edges

def create_iupac_condensed(nodes, edges):
    edge_marker = 1
    cyclo = False
    amino_acids = nodes[:]
    for i, j in edges:
        if min(i, j) == 0 and max(i, j) == len(amino_acids)-1:
            cyclo = True
            continue
        amino_acids[i] += f'({edge_marker})'
        amino_acids[j] += f'({edge_marker})'
        edge_marker += 1
    sequence = '-'.join(amino_acids)
    sequence = f'cyclo[{sequence}]' if cyclo else sequence
    return sequence

def create_amino_acid_chain(nodes, edges):
    edge_marker = 1
    amino_acids = nodes[:]
    for i, j in edges:
        amino_acids[i] += f'({edge_marker})'
        amino_acids[j] += f'({edge_marker})'
        edge_marker += 1
    sequence = '--'.join(amino_acids)
    return sequence

def create_graph_presentation(nodes, edges):
    sequence = ','.join(nodes)
    edges_pre = '' if len(edges)==0 else ' '.join(['@'+','.join([str(k) for k in pair]) for pair in edges])
    return (sequence +' '+edges_pre).strip()

def create_one_character_peptide(nodes): # 忽略侧链互作
    amino_acid_refs = {i:j for j, i, _, _ in AminoAcids}
    amino_acid_refs['UNK'] = 'X'
    tail = ''
    if '(NH2)' == nodes[-1].upper()[-5:]:
        nodes[-1] = nodes[-1][:-5]
        tail = '(NH2)'
    is_essential_aa = [i.upper().strip() in amino_acid_refs.keys() for i in nodes]
    if False in is_essential_aa:
        return None
    else:
        return ''.join([amino_acid_refs[i.upper().strip()] for i in nodes])+tail

def read_one_character_sequence(sequence):
    tail = ''
    if '(NH2)' == sequence.upper()[-5:]:
        sequence = sequence[:-5]
        tail = '(NH2)'
    amino_acid_refs = {i:j for i, j, _, _ in AminoAcids}
    amino_acid_refs['X'] = 'UNK'
    aas = [amino_acid_refs[i] for i in sequence]
    aas[-1] = aas[-1]+tail
    return aas, []

def read_graph_representation(sequence):
    amin_acids = [i.strip() for i in sequence.split(' @')[0].strip().split(',')]
    edges = [tuple([int(j) for j in i.strip().split(',')]) for i in sequence.split('@')[1:]]
    return amin_acids, edges

def read_sequence(sequence):
    seq_format = ''
    if '@' in sequence:
        seq_format = 'Graph presentation'
        nodes, edges = read_graph_representation(sequence)
    elif ',' in sequence:
        no_parentheses = re.sub(r'\([^)]*\)', '', sequence) # 删除所有括号中的逗号
        if ',' in no_parentheses:
            seq_format = 'Graph presentation'
            nodes, edges = read_graph_representation(sequence)
        else:
            seq_format = 'IUPAC condensed'
            nodes, edges = read_iupac_condensed(sequence, sep='-')
    elif '--' in sequence:
        seq_format = 'Amino acid chain'
        nodes, edges = read_iupac_condensed(sequence, sep='--')
    elif '-' in sequence:
        seq_format = 'IUPAC condensed'
        nodes, edges = read_iupac_condensed(sequence, sep='-')
    else:
        seq_format = 'One character peptide'
        nodes, edges = read_one_character_sequence(sequence)
    return seq_format, nodes, edges
        
def create_sequence(nodes, edges):
    try:
        iupac_condensed = create_iupac_condensed(nodes, edges)
    except:
        iupac_condensed = None
    try:
        amino_acid_chain = create_amino_acid_chain(nodes, edges)
    except:
        amino_acid_chain = None
    try:
        graph_presentation = create_graph_presentation(nodes, edges)
    except:
        graph_presentation = None
    try:
        one_character_peptide = create_one_character_peptide(nodes)
    except:
        one_character_peptide = None
    return {'iupac_condensed':iupac_condensed, 'amino_acid_chain':amino_acid_chain, 
            'graph_presentation':graph_presentation, 'one_character_peptide':one_character_peptide}


