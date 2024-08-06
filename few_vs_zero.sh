
#python src/activate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --shot 5 
#python src/identify.py --task task274_overruling_legal_classification.json --mod GV_trace-test --shot 5


python src/evaluate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --mask_shot 5 --task_shot 0 --multiplier 0.2
python src/evaluate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --mask_shot 5 --task_shot 0 --multiplier 0.4
python src/evaluate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --mask_shot 5 --task_shot 0 --multiplier 0.6
python src/evaluate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --mask_shot 5 --task_shot 0 --multiplier 0.8
#python src/evaluate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --mask_shot 5 --task_shot 0 --multiplier 1.0
#python src/evaluate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --mask_shot 5 --task_shot 0 --multiplier 1.2
#python src/evaluate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --mask_shot 5 --task_shot 0 --multiplier 1.4
#python src/evaluate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --mask_shot 5 --task_shot 0 --multiplier 1.6
#python src/evaluate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --mask_shot 5 --task_shot 0 --multiplier 1.8
#python src/evaluate.py --task task274_overruling_legal_classification.json --mod GV_trace-test --mask_shot 5 --task_shot 0 --multiplier 2.0
