# GitHub release checklist

1. Verify the intended owner/author list and add the approved LICENSE and any
   required NOTICE text. Check source, dependencies, model weights and data terms.
2. Read MIGRATION.md and TEST_REPORT.md. Keep rc1 until target-environment checks
   and held-out data evaluation have been reviewed. Do not label it stable solely
   because the CPU smoke tests pass.
3. Create a new virtual environment, install the selected extras, and run
   `python -m pytest -q`. For the primary model, run
   `eddy-doctor --architecture deeplabv3plus_resnet101 --num_classes 3 --forward`.
4. Configure the exact mask legend and train/validation/test scene grouping.
   Confirm that ignore codes are not foreground classes, and that zero handling
   matches the chosen raw or CLAHE-preprocessed imagery.
5. Confirm that no imagery, weights, local secrets, notebook outputs or private
   paths are staged. The included notebook is unexecuted and .gitignore excludes
   common datasets/checkpoints. Review any additional files before committing.
6. Initialize the local Git repository and push to an empty GitHub repository
   using the commands in README.md. The generated archive has not performed
   any remote actions or created a release/tag on your behalf.
7. Run GitHub Actions and your GPU checks. Record actual dependency versions and
   the held-out evaluation protocol before publishing trained checkpoints.

The file `legacy/train_ocean_eddy_aug_level.py` remains the original designated
stable script. `.gitattributes` and the manifest test protect its bytes.
