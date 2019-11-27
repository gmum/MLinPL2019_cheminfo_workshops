import os


def _refine_pdbqt(infile):
    lines = []
    with open(infile, 'r') as stream:
        for line in stream:
            lines.append(line)
    with open(infile, 'w') as output:
        for line in lines:
            if 'ROOT' in line or 'BRANCH' in line or 'TORSDOF' in line:
                continue
            else:
                output.write(line)


def dock_smiles(smi, receptor, pocket_center, ligand_name):
    """
Dock a compound to the provided protein
    :param smi: str; SMILES representation of a molecule
    :param receptor: str; Path to the .pdb file of the protein
    :param pocket_center: tuple of floats; (x,y,z) coordinates of the binding pocket center
    :param ligand_name: str; Name of the ligand to be included in output name
    :return: float; Binding energy of the complex (kcal/mol)
    """
    basepath, name = os.path.split(receptor)
    ligand = os.path.join(basepath, "{}.smi".format(ligand_name))
    with open(ligand, "w") as outfile:
        outfile.write(smi)
    lig_pdbqt = ligand.replace(".smi", ".pdbqt")
    receptor_pdbqt = receptor.replace(".pdb", ".pdbqt")
    if not os.path.exists(receptor_pdbqt):
        os.system('babel -ipdb {receptor} -opdbqt {output}'.format(receptor=receptor, output=receptor_pdbqt))
        _refine_pdbqt(receptor_pdbqt)
    receptor = receptor_pdbqt
    os.system('babel -ismi {ligand} -opdbqt {output} --gen3d -p 7.4 >> babel.log'.format(
        ligand=ligand, output=lig_pdbqt))
    output_name = "{}_{}.pdbqt".format(receptor.replace(".pdbqt", ""), ligand_name)
    os.system('vina --receptor {receptor} \
    --ligand {ligand} \
    --center_x {x} \
    --center_y {y} \
    --center_z {z} \
    --size_x {size} \
    --size_y {size} \
    --size_z {size} \
    --out {output}'.format(
        receptor=receptor,
        ligand=lig_pdbqt,
        x=pocket_center[0],
        y=pocket_center[1],
        z=pocket_center[2],
        size=15,
        output=output_name
    ))
    os.system('babel -ipdbqt {input} -osdf {output} -p'.format(input=output_name, output=output_name.replace('.pdbqt', '.sdf')))
    dock_energy = _read_dock_energy_from_sdf(output_name.replace('.pdbqt', '.sdf'))
    return dock_energy


def _read_dock_energy_from_sdf(sdf_file):
    dock_energy = None
    with open(sdf_file, "r") as result:
        for line in result:
            if "VINA RESULT:" in line:
                dock_energy = float(line.split()[2])
                break
    return dock_energy
