# rp2-with-cam
combining with cam to crafte mask
首先下载https://github.com/evtimovi/robust_physical_perturbations 仓库的gtsrb-cnn-attack的代码,按照robust_physical_perturbation的环境要求进行环境配置
使用本仓库的cleverhans和classify_yadav.py替换gtsrb-cnn-attack中的cleverhans和classify_yadav.py
根据cam得到mask:
      执行run_classify.sh
