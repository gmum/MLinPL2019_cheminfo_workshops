import os, sys, math
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import rdmolops
import pandas as pd
from Bio import PDB
import argparse
from scipy.spatial import distance_matrix


def get_residues_and_atoms(pdbfile):
	parser = PDB.PDBParser()
	residue_names = {}
	residue_atoms = {}
	structure = parser.get_structure(file=pdbfile, id="0")
	for chain in structure[0]:
		for residue in chain:
			res_id = residue.get_id()
			if res_id[0] != " ":
				continue
			resname = residue.get_resname()
			residue_names[res_id[1]] = resname
			residue_atoms[res_id[1]] = {}
			for atom in residue:
				atom_id = atom.get_id()
				atom_coords = atom.get_coord()
				residue_atoms[res_id[1]][atom_id] = atom_coords
	return residue_names, residue_atoms


def get_compounds(sdffile):
	df = PandasTools.LoadSDF(sdffile)
	molecules = []
	for mol in df["ROMol"]:
		molecules.append(mol)
	return molecules


def get_compound_atom_coords(mol):
	coords = []
	conf = mol.GetConformer()
	for atom in mol.GetAtoms():
		coords.append(conf.GetAtomPosition(atom.GetIdx()))
	return coords


def calculate_angle(atom1,atom2,atom3):
	vector1 = atom1 - atom2 #(atom1[0]-atom2[0],atom1[1]-atom2[1],atom1[2]-atom2[2])
	vector2 = atom3 - atom2 #(atom3[0]-atom2[0],atom3[1]-atom2[1],atom3[2]-atom2[2])
	v1mag = numpy.linalg.norm(vector1) # np.srqt(np.sum(vector1*vector1))
	v1norm = vector1 / v1mag # (vector1[0]/v1mag,vector1[1]/v1mag,vector1[2]/v1mag)
	v2mag = numpy.linalg.norm(vector2) # math.sqrt(vector2[0]*vector2[0] + vector2[1]*vector2[1] + vector2[2]*vector2[2])
	v2norm = vector2 / v2mag # (vector2[0]/v2mag,vector2[1]/v2mag,vector2[2]/v2mag)
	res = np.sum(v1norm * v2norm) # v1norm[0]*v2norm[0] + v1norm[1]*v2norm[1] + v1norm[2]*v2norm[2]
	angle = math.acos(res)
	angle = math.degrees(angle)
	return angle


class SIFt_generator:
	min_donor_angle = 120
	min_acc_angle = 90
	Any = 1
	Backbone = 2
	Sidechain = 3
	Polar = 4
	Hydrophobic = 5
	HB_donor = 6
	HB_acceptor = 7
	Aromatic = 8
	Charged = 9
	Halogen = 10
	HBOND_DONOR_RESIDUES = ["ARG", "HIS", "LYS", "ASN", "GLN", "CYS", "TYR"]
	HBOND_ACCEPTOR_RESIDUES = ["ASP", "GLU", "SER", "THR", "ASN", "GLN", "TYR", "HIS"]
	CHARGED_RESIDUES = ["ARG", "ASP", "GLU", "LYS", "HIS", "CYT"]
	AROMATIC_RESIDUES = ["PHE", "TYR", "TRP"]
	POLAR_RESIDUES = ["ARG", "ASP", "GLU", "HIS", "ASN", "GLN", "LYS", "SER", "THR"]
	HYDROPHOBIC_RESIDUES = ["PHE", "LEU", "ILE", "TYR", "TRP", "VAL", "MET", "PRO", "CYS", "ALA"]
	hbond_acceptors = ["O", "N", "S"]

	def __init__(self, cutoff=4.0, halogen=False, extended=False):
		self.ext = extended
		self.cutoff = float(cutoff)
		self.halogen = halogen

	def get_all_interactions(self, pdbfile, sdffile):
		residue_names, residue_atoms = get_residues_and_atoms(pdbfile)
		compounds = get_compounds(sdffile)
		all_interactions = {}
		for index, compound in enumerate(compounds):
			compound = rdmolops.AddHs(compound)
			interactions = self.get_interactions(compound, residue_atoms, residue_names)
			all_interactions[index] = interactions
		return all_interactions

	def get_interactions(self, compound, residue_atoms, residue_names):
		valid_residues = []
		compound_coords = get_compound_atom_coords(compound)
		for residue_num in residue_atoms:
			res_atom_coords = [a for a in residue_atoms[residue_num].values()]
			distances = distance_matrix(res_atom_coords, compound_coords)
			if np.min(distances) <= self.cutoff:
				valid_residues.append(residue_num)

		interactions = {}
		for residue_num in residue_atoms:
			if residue_num not in valid_residues:
				interactions[residue_num] = [0] * 9
				continue
			res_interactions = [
				1,
				self.get_backbone_interaction(residue_atoms[residue_num], compound_coords),
				self.get_sidechain_interaction(residue_atoms[residue_num], compound_coords),
				self.get_polar_interaction(residue_names[residue_num]),
				self.get_hydrophobic_interaction(residue_names[residue_num]),
				self.get_hbond_donors(compound, residue_names[residue_num], residue_atoms[residue_num]),
				self.get_hbond_acceptors(compound, residue_names[residue_num], residue_atoms[residue_num]),
				self.get_aromatic_interaction(residue_names[residue_num]),
				self.get_charged_interaction(residue_names[residue_num])
				]
			interactions[residue_num] = res_interactions
		return interactions

	def get_backbone_interaction(self, residue, compound_coords):
		backbone_atoms = [residue[atom] for atom in ["C", "CA", "N", "O"]]
		distances = distance_matrix(backbone_atoms, compound_coords)
		if np.min(distances) <= self.cutoff:
			return 1
		return 0

	def get_sidechain_interaction(self, residue, compound_coords):
		backbone_atoms = [residue[atom] for atom in residue if atom not in ["C", "CA", "N", "O"]]
		distances = distance_matrix(backbone_atoms, compound_coords)
		if np.min(distances) <= self.cutoff:
			return 1
		return 0

	def get_polar_interaction(self, residue_name):
		if residue_name in self.POLAR_RESIDUES:
			return 1
		return 0

	def get_hydrophobic_interaction(self, residue_name):
		if residue_name in self.HYDROPHOBIC_RESIDUES:
			return 1
		return 0

	def get_aromatic_interaction(self, residue_name):
		if residue_name in self.AROMATIC_RESIDUES:
			return 1
		return 0

	def get_charged_interaction(self, residue_name):
		if residue_name in self.CHARGED_RESIDUES:
			return 1
		return 0

	def get_hbond_donors(self, compound, residue_name, residue_atoms):
		if residue_name not in self.HBOND_DONOR_RESIDUES:
			return 0
		if residue_name in ["ARG", "ARN"]:
			hbond_donors = [residue_atoms[atom] for atom in ["HH11", "HH12", "HH21", "HH22", "HE"] if atom in residue_atoms]
		if residue_name == "HIS":
			hbond_donors = [residue_atoms[atom] for atom in ["HE2"] if atom in residue_atoms]
		if residue_name == "LYS":
			hbond_donors = [residue_atoms[atom] for atom in ["HZ1", "HZ2", "HZ3"] if atom in residue_atoms]
		if residue_name == "ASN":
			hbond_donors = [residue_atoms[atom] for atom in ["HD21", "HD22"] if atom in residue_atoms]
		if residue_name == "GLN":
			hbond_donors = [residue_atoms[atom] for atom in ["HE22", "HE21"] if atom in residue_atoms]
		if residue_name == "TYR":
			hbond_donors = [residue_atoms[atom] for atom in ["HH"] if atom in residue_atoms]
		if residue_name in ["CYS", "CYX", "SER"]:
			hbond_donors = [residue_atoms[atom] for atom in ["HG"] if atom in residue_atoms]
		if residue_name == "THR":
			hbond_donors = [residue_atoms[atom] for atom in ["HG1"] if atom in residue_atoms]
		if hbond_donors == []:
			return 0
		hbond_acceptors = [atom for atom in compound.GetAtoms() if atom.GetSymbol() in self.hbond_acceptors]
		conf = compound.GetConformer()
		hbond_acceptors_coords = [conf.GetAtomPosition(atom.GetIdx()) for atom in hbond_acceptors]
		distances = distance_matrix(hbond_acceptors_coords, hbond_donors)
		close_atoms = np.min(distances, axis=1) <= 2.5
		if np.sum(close_atoms) == 0:
			return 0
		for index, distance in enumerate(distances[close_atoms]):
			close_donors = distance <= 2.5
			acceptor = hbond_acceptors[close_atoms][index]
			for donor in hbond_donors[close_donors]:
				acceptor_base = [neighbor for neighbor in acceptor.GetNeighbors() if neighbor.GetSymbol() != "H"][0]
				angle = calculate_angle(conf.GetAtomPosition(acceptor_base.GetIdx()), conf.GetAtomPosition(acceptor.GetIdx()), donor)
				if angle >= self.min_donor_angle:
					return 1
				else:
					print("else")
		return 0

	def get_hbond_acceptors(self, compound, residue_name, residue_atoms):
		if residue_name not in self.HBOND_ACCEPTOR_RESIDUES:
			return 0
		if residue_name == "ASP":
			hbond_acceptors = [residue_atoms[atom] for atom in ["OD1", "OD2"] if atom in residue_atoms]
			acceptor_base = 'CG'
		if residue_name == "GLU":
			hbond_acceptors = [residue_atoms[atom] for atom in ["OE1", "OE2"] if atom in residue_atoms]
			acceptor_base = 'CD'
		if residue_name == "SER":
			hbond_acceptors = [residue_atoms[atom] for atom in ["OG"] if atom in residue_atoms]
			acceptor_base = 'CB'
		if residue_name == "THR":
			hbond_acceptors = [residue_atoms[atom] for atom in ["OG1"] if atom in residue_atoms]
			acceptor_base = 'CB'
		if residue_name == "ASN":
			hbond_acceptors = [residue_atoms[atom] for atom in ["OD1", "ND2"] if atom in residue_atoms]
			acceptor_base = 'CG'
		if residue_name == "GLN":
			hbond_acceptors = [residue_atoms[atom] for atom in ["OE1", "NE2"] if atom in residue_atoms]
			acceptor_base = 'CD'
		if residue_name == "TYR":
			hbond_acceptors = [residue_atoms[atom] for atom in ["OH"] if atom in residue_atoms]
			acceptor_base = 'CZ'
		if residue_name == "HIS":
			hbond_acceptors = [residue_atoms[atom] for atom in ["ND1", "NE2"] if atom in residue_atoms]
			acceptor_base = 'CE1'
		acceptor_base = residue_atoms[acceptor_base]
		conf = compound.GetConformer()
		hbond_donors = [conf.GetAtomPosition(atom.GetIdx()) for atom in compound.GetAtoms() if atom.GetSymbol() == "H"
						and atom.GetNeighbors()[0].GetSymbol() in self.hbond_acceptors]

		distances = distance_matrix(hbond_acceptors, hbond_donors)
		close_atoms = np.min(distances, axis=1) <= 2.5
		if np.sum(close_atoms) == 0:
			return 0
		for index, distance in enumerate(distances[close_atoms]):
			close_donors = distance <= 2.5
			acceptor = hbond_acceptors[close_atoms][index]
			for donor in hbond_donors[close_donors]:
				angle = calculate_angle(acceptor_base, acceptor, donor)
				if angle >= self.min_donor_angle:
					return 1
				else:
					print("else")
		return 0
