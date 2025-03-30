.PHONY: hello container hpc_normalize hpc_ci_net sweep run lrp mean_std test cbir lr_finder k_fold_cross_validation grad_cam

hello:
	echo "Hello World!"

container:
	chmod u+x containers/create_container.sh
	./containers/create_container.sh

hpc_normalize:
	chmod u+x src/scripts/ham_normalize.sh
	sbatch src/scripts/ham_normalize.sh

hpc_ci_net:
	sbatch src/scripts/ci_net.sh

sweep:
	sbatch src/scripts/sweep.sh

run:
	sbatch src/scripts/run.sh

lrp:
	sbatch src/scripts/lrp.sh

mean_std:
	sbatch src/scripts/mean_std.sh

test:
	sbatch src/scripts/test.sh

cbir:
	sbatch src/scripts/cbir.sh

lr_finder:
	sbatch src/scripts/lr_finder.sh

k_fold_cross_validation:
	sbatch src/scripts/k_fold_cross_validation.sh

grad_cam:
	sbatch src/scripts/grad_cam.sh