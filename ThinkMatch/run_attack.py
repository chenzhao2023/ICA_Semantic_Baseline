f = open("run_attack.sh", "w")
# run NGM
for cv in range(5):
    for prob in [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]:
        cmd = "python train_artery_ngm_feat.py --flow=attack --prob={} --exp=exp/ngm/CV{} --cv={} --plot=True".format(prob, cv, cv)
        print(cmd)
        f.write(f"{cmd}\n")


# run PCA
for cv in range(5):
    for prob in [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]:
        cmd = "python train_artery_pca.py --flow=attack --prob={} --exp=exp/pca/CV{} --cv={} --plot=True".format(prob, cv, cv)
        print(cmd)
        f.write(f"{cmd}\n")





f.flush()
f.close()



