```
print('now testing with individual epochs and learning rate of 5e-2')
num_cycles = 4
prev_cycles = 4

print('This model has been trained for', prev_cycles, 'epochs already')    
file = lm_base_file + str(prev_cycles)
learner_file = base_path/file
learn.load(learner_file)
learn.unfreeze()
print('loaded existing learner from', str(learner_file))

for n in range(num_cycles):
    learn.fit_one_cycle(1, 5e-3, moms=(0.8,0.7))
    print('    ', n + 1, 'addtional run of fit_one_cycle complete')
    file = lm_base_file + str(prev_cycles + n + 1)
    learner_file = base_path/file
    learn.save(learner_file)
    release_mem()
    
print('completed', num_cycles, 'new training epochs')
print('completed', num_cycles + prev_cycles, 'total training epochs')
```

	now testing with individual epochs and learning rate of 5e-2
	This model has been trained for 4 epochs already
	loaded existing learner from /home/seth/mimic/mimic_lm_fine_tuned_4
	
	epoch 	train_loss 	valid_loss 	accuracy 	time
	0 	1.742496 	1.850268 	0.624424 	12:58
	
	     1 addtional run of fit_one_cycle complete
	
	epoch 	train_loss 	valid_loss 	accuracy 	time
	0 	1.798283 	1.839677 	0.625925 	12:57
	
	     2 addtional run of fit_one_cycle complete
	
	epoch 	train_loss 	valid_loss 	accuracy 	time
	0 	1.779387 	1.832521 	0.626880 	12:57
	
	     3 addtional run of fit_one_cycle complete
	
	epoch 	train_loss 	valid_loss 	accuracy 	time
	0 	1.759929 	1.827754 	0.627306 	13:01
	
	     4 addtional run of fit_one_cycle complete
	completed 4 new training epochs
	completed 8 total training epochs





```
print('now testing with multiple epochs and learning rate of 5e-2')
num_cycles = 4
prev_cycles = 4


print('This model has been trained for', prev_cycles, 'epochs already')    
file = lm_base_file + str(prev_cycles)
learner_file = base_path/file
learn.load(learner_file)
learn.unfreeze()
print('loaded existing learner from', str(learner_file))


learn.fit_one_cycle(num_cycles, 5e-3, moms=(0.8,0.7))
file = lm_base_file + str(prev_cycles + num_cycles + 1)
learner_file = base_path/file
learn.save(learner_file)
release_mem()
    
print('completed', num_cycles, 'new training epochs')
print('completed', num_cycles + prev_cycles, 'total training epochs')
```

	now testing with multiple epochs and learning rate of 5e-2
	This model has been trained for 4 epochs already
	loaded existing learner from /home/seth/mimic/mimic_lm_fine_tuned_4
	
	epoch 	train_loss 	valid_loss 	accuracy 	time
	0 	1.945363 	1.966063 	0.604700 	12:57
	1 	1.937261 	1.932958 	0.610114 	12:57
	2 	1.773702 	1.830366 	0.626390 	12:57
	3 	1.657113 	1.798879 	0.632739 	12:57
	
	completed 4 new training epochs
	completed 8 total training epochs


	now testing with individual epochs and learning rate of 1e-2
	This model has been trained for 4 epochs already
	loaded existing learner from /home/seth/mimic/mimic_lm_fine_tuned_4
	
	epoch 	train_loss 	valid_loss 	accuracy 	time
	0 	1.997753 	1.940282 	0.611250 	12:59
	
	     1 addtional run of fit_one_cycle complete
	
	epoch 	train_loss 	valid_loss 	accuracy 	time
	0 	1.982868 	1.933895 	0.611702 	12:57
	
	     2 addtional run of fit_one_cycle complete
	
	epoch 	train_loss 	valid_loss 	accuracy 	time
	0 	1.946961 	1.921856 	0.613163 	12:57
	
	     3 addtional run of fit_one_cycle complete
	
	epoch 	train_loss 	valid_loss 	accuracy 	time
	0 	1.884777 	1.911755 	0.614383 	12:56



	now testing with multiple epochs and learning rate of 1e-2
	This model has been trained for 4 epochs already
	loaded existing learner from /home/seth/mimic/mimic_lm_fine_tuned_4
	
	epoch 	train_loss 	valid_loss 	accuracy 	time
	0 	2.199889 	2.164528 	0.575957 	12:56
	1 	2.133318 	2.114752 	0.583654 	12:57
	2 	1.927248 	1.924726 	0.610823 	12:57
	3 	1.773408 	1.856625 	0.622632 	12:59
	
	completed 4 new training epochs
	completed 8 total training epochs
