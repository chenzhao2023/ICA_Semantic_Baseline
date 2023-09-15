
# GENERATE COMMANDS
from itertools import islice
 

GPU_IDS = [0,0,0,0,0,0]


def seperate_commands_to_sub_files(commands, n_gpus=4, prefix="run"):
    batch_size = len(commands)//n_gpus
    if len(commands) % n_gpus == 0:
        length_to_split = [batch_size]*n_gpus
    else:
        length_to_split = [batch_size]*(n_gpus-1)
        length_to_split.append(len(commands) - batch_size*(n_gpus-1))
    
    inputt = iter(commands)
    output = [list(islice(inputt, elem)) for elem in length_to_split]

    for idx, l in enumerate(output):
        f = open(f"{prefix}_{idx}.sh", "w")
        for c in l:
            f.write(f"{c} --gpu={GPU_IDS[idx]}\n")
        f.flush()
        f.close()


if __name__ == '__main__':

    commands = []
    for cv in range(5):
        exp_path = f"exp/ngm/CV{cv}"
        command = f"python train_artery_ngm_feat.py --exp={exp_path} --cv={cv}"
        commands.append(command)
    seperate_commands_to_sub_files(commands, 1, f"run_ngm")
    print(len(commands))

    commands = []
    for cv in range(5):
        exp_path = f"exp/pca/CV{cv}"
        command = f"python train_artery_pca.py --exp={exp_path} --cv={cv}"
        commands.append(command)
    seperate_commands_to_sub_files(commands, 1, f"run_pca")
    print(len(commands))

