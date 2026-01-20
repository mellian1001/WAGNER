RULES_PATH='temp_result/human_designed_rules.txt'
SAVE_PLAN_PATH='temp_result/plan1.txt'
SAVE_GOAL_LIST_PATH='temp_result/goal_list1.txt'
SAVE_WRAPPERS_PATH='temp_result/submodel_wrappers1.py'
SAVE_MODEL_INFO_PATH='temp_result/submodels1.json'

FINAL_TASK='collect an iron'


python planning.py \
    --rules_path "$RULES_PATH" \
    --final_task "$FINAL_TASK" \
    --save_plan_path "$SAVE_PLAN_PATH" \
    --save_goal_list_path "$SAVE_GOAL_LIST_PATH" 

python propose_subRLs.py \
    --rules_path "$RULES_PATH" \
    --plan_path "$SAVE_PLAN_PATH" \
    --save_model_info_path "$SAVE_MODEL_INFO_PATH" \
    --final_task "$FINAL_TASK" \
    --save_wrappers_path "$SAVE_WRAPPERS_PATH"

