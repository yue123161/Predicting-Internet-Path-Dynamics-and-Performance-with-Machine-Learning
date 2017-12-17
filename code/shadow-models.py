from utils import *
model = 'XGB'
# model = 'RF'
task = 1

if model=='RF' and task==1:
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task1/task1_70+30/70/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task1/task1_70+30/30/',
            task=1,
            task_name='task1/data-1-task-1-7030-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task1/task1_80+20/80/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task1/task1_80+20/20/',
            task=1,
            task_name='task1/data-1-task-1-8020-RF',
            n_estimators=10)

    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task1/task1_70+30/70/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task1/task1_70+30/30/',
            task=1,
            task_name='task1/data-2-task-1-7030-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task1/task1_80+20/80/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task1/task1_80+20/20/',
            task=1,
            task_name='task1/data-2-task-1-8020-RF',
            n_estimators=10)

    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/70/windowsize10/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/30/windowsize10/',
            task=1,
            task_name='task1/data-3-task-1-7030-windowsize10-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/70/windowsize20/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/30/windowsize20/',
            task=1,
            task_name='task1/data-3-task-1-7030-windowsize20-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/70/windowsize50/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/30/windowsize50/',
            task=1,
            task_name='task1/data-3-task-1-7030-windowsize50-RF',
            n_estimators=10)

    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/80/windowsize10/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/20/windowsize10/',
            task=1,
            task_name='task1/data-3-task-1-8020-windowsize10-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/80/windowsize20/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/20/windowsize20/',
            task=1,
            task_name='task1/data-3-task-1-8020-windowsize20-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/80/windowsize50/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/20/windowsize50/',
            task=1,
            task_name='task1/data-3-task-1-8020-windowsize50-RF',
            n_estimators=10)

########################################################################################################
if model=='RF' and task==3:
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task3/task3_70+30/70/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task3/task3_70+30/30/',
            task=3,
            task_name='task3/data-1-task-3-7030-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task3/task3_80+20/80/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task3/task3_80+20/20/',
            task=3,
            task_name='task3/data-1-task-3-8020-RF',
            n_estimators=10)

    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task3/task3_70+30/70/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task3/task3_70+30/30/',
            task=3,
            task_name='task3/data-2-task-3-7030-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task3/task3_80+20/80/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task3/task3_80+20/20/',
            task=3,
            task_name='task3/data-2-task-3-8020-RF',
            n_estimators=10)

    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/70/windowsize10/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/30/windowsize10/',
            task=3,
            task_name='task3/data-3-task-3-7030-windowsize10-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/70/windowsize20/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/30/windowsize20/',
            task=3,
            task_name='task3/data-3-task-3-7030-windowsize20-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/70/windowsize50/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/30/windowsize50/',
            task=3,
            task_name='task3/data-3-task-3-7030-windowsize50-RF',
            n_estimators=10)

    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/80/windowsize10/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/20/windowsize10/',
            task=3,
            task_name='task3/data-3-task-3-8020-windowsize10-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/80/windowsize20/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/20/windowsize20/',
            task=3,
            task_name='task3/data-3-task-3-8020-windowsize20-RF',
            n_estimators=10)
    RF_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/80/windowsize50/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/20/windowsize50/',
            task=3,
            task_name='task3/data-3-task-3-8020-windowsize50-RF',
            n_estimators=10)

####################################################################################################
if model=='XGB' and task==1:
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task1/task1_70+30/70/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task1/task1_70+30/30/',
            task=1,
            task_name='task1-XGB/data-1-task-1-7030-XGB',
            n_estimators=100)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task1/task1_80+20/80/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task1/task1_80+20/20/',
            task=1,
            task_name='task1-XGB/data-1-task-1-8020-XGB',
            n_estimators=100)

    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task1/task1_70+30/70/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task1/task1_70+30/30/',
            task=1,
            task_name='task1-XGB/data-2-task-1-7030-XGB',
            n_estimators=100)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task1/task1_80+20/80/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task1/task1_80+20/20/',
            task=1,
            task_name='task1-XGB/data-2-task-1-8020-XGB',
            n_estimators=100)

    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/70/windowsize10/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/30/windowsize10/',
            task=1,
            task_name='task1-XGB/data-3-task-1-7030-windowsize10-XGB',
            n_estimators=100)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/70/windowsize20/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/30/windowsize20/',
            task=1,
            task_name='task1-XGB/data-3-task-1-7030-windowsize20-XGB',
            n_estimators=100)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/70/windowsize50/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_70+30/30/windowsize50/',
            task=1,
            task_name='task1-XGB/data-3-task-1-7030-windowsize50-XGB',
            n_estimators=100)

    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/80/windowsize10/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/20/windowsize10/',
            task=1,
            task_name='task1-XGB/data-3-task-1-8020-windowsize10-XGB',
            n_estimators=100)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/80/windowsize20/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/20/windowsize20/',
            task=1,
            task_name='task1-XGB/data-3-task-1-8020-windowsize20-XGB',
            n_estimators=100)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/80/windowsize50/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task1/task1_80+20/20/windowsize50/',
            task=1,
            task_name='task1-XGB/data-3-task-1-8020-windowsize50-XGB',
            n_estimators=100)

###############################################################################################################
if model=='XGB' and task==3:
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task3/task3_70+30/70/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task3/task3_70+30/30/',
            task=3,
            task_name='task3-XGB/data-1-task-3-7030-XGB',
            n_estimators=10)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task3/task3_80+20/80/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/1/task3/task3_80+20/20/',
            task=3,
            task_name='task3-XGB/data-1-task-3-8020-XGB',
            n_estimators=10)

    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task3/task3_70+30/70/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task3/task3_70+30/30/',
            task=3,
            task_name='task3-XGB/data-2-task-3-7030-XGB',
            n_estimators=10)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task3/task3_80+20/80/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/2/task3/task3_80+20/20/',
            task=3,
            task_name='task3-XGB/data-2-task-3-8020-XGB',
            n_estimators=10)

    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/70/windowsize10/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/30/windowsize10/',
            task=3,
            task_name='task3-XGB/data-3-task-3-7030-windowsize10-XGB',
            n_estimators=10)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/70/windowsize20/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/30/windowsize20/',
            task=3,
            task_name='task3-XGB/data-3-task-3-7030-windowsize20-XGB',
            n_estimators=10)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/70/windowsize50/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_70+30/30/windowsize50/',
            task=3,
            task_name='task3-XGB/data-3-task-3-7030-windowsize50-XGB',
            n_estimators=10)

    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/80/windowsize10/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/20/windowsize10/',
            task=3,
            task_name='task3-XGB/data-3-task-3-8020-windowsize10-XGB',
            n_estimators=10)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/80/windowsize20/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/20/windowsize20/',
            task=3,
            task_name='task3-XGB/data-3-task-3-8020-windowsize20-XGB',
            n_estimators=10)
    XGB_task(train_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/80/windowsize50/',
            test_path='/home/fw/Documents/CN-group-project/data/temp_data/3/task3/task3_80+20/20/windowsize50/',
            task=3,
            task_name='task3-XGB/data-3-task-3-8020-windowsize50-XGB',
            n_estimators=10)