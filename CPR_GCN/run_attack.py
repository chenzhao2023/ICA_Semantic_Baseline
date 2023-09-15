
f = open("run_attack.sh", "w")
for cv in range(5):
    for prob in [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]:
        cmd = "python train_gpr_gcn.py --exp=exp/MLP64_GCN128/CV{} --cv={} --cv_max=5 --device=cuda:0 --train=attack --prob={}".format(cv, cv, prob)
        print(cmd)
        f.write(f"{cmd}\n")
    
f.flush()
f.close()