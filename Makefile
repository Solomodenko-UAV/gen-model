run_on_cpu:
	echo "Running on CPU"
	USE_GPU=False python3 main.py

run_on_gpu:
	echo "Running on GPU"
	set USE_GPU=True && python main.py