.PHONY: hello singularity_container enroot_container sweep run lrp augmentation_viewer test cbir lr_finder k_fold_cross_validation grad_cam

hello:
	echo "Hello World!"

singularity_container:
	chmod u+x containers/create_singularity_container.sh
	./containers/create_singularity_container.sh

enroot_container:
	chmod u+x containers/create_enroot_container.sh
	./containers/create_enroot_container.sh

sweep:
	sbatch src/scripts/sweep.sh

run:
	sbatch src/scripts/run.sh

lrp:
	sbatch src/scripts/lrp.sh

augmentation_viewer:
	sbatch src/scripts/augmentation_viewer.sh

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