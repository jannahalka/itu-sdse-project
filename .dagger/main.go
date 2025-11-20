// A generated module for MlPipeline functions
//
// This module has been generated via dagger init and serves as a reference to
// basic module structure as you get started with Dagger.
//
// Two functions have been pre-created. You can modify, delete, or add to them,
// as needed. They demonstrate usage of arguments and return types using simple
// echo and grep commands. The functions can be called from the dagger CLI or
// from one of the SDKs.
//
// The first line in this comment block is a short description line and the
// rest is a long description with more detail on the module's purpose or usage,
// if appropriate. All modules should have a short description.

package main

import (
	"context"
	"dagger/ml-pipeline/internal/dagger"
)

type MlPipeline struct{}

func (m *MlPipeline) Download(
	ctx context.Context,
	// +defaultPath="/"
	src *dagger.Directory,
) *dagger.File {
	container := m.Select(ctx, src).
		WithExec([]string{"mlflow", "artifacts", "download", "-u", "models:/model@staging", "-d", "/tmp/model"}).
		WithExec([]string{"bash", "-c", "cp /tmp/model/artifacts/*.pkl /tmp/model.pkl"})
	return container.File("/tmp/model.pkl")
}

func (m *MlPipeline) Select(
	ctx context.Context,
	// +defaultPath="/"
	src *dagger.Directory,
) *dagger.Container {
	return m.Train(src).
		WithExec([]string{"python", "itu_sdse_project/modeling/selection.py"})
}

func (m *MlPipeline) Train(
	// +defaultPath="/"
	src *dagger.Directory,
) *dagger.Container {
	return m.PrepareData(src).
		WithExec([]string{"python", "itu_sdse_project/modeling/train.py", "xgboost"}).
		WithExec([]string{"python", "itu_sdse_project/modeling/train.py", "log-reg"})
}

func (m *MlPipeline) PrepareData(
	// +defaultPath="/"
	src *dagger.Directory,
) *dagger.Container {
	return m.BuildEnv(src).
		WithExec([]string{"python", "data/interim/make_dataset.py"}).
		WithExec([]string{"python", "itu_sdse_project/features.py"})
}

func (m *MlPipeline) Test(
	ctx context.Context,
	// +defaultPath="/"
	src *dagger.Directory,
) (string, error) {
	return m.BuildEnv(src).
		WithExec([]string{"pytest"}).
		Stdout(ctx)
}

func (m *MlPipeline) BuildEnv(
	// +defaultPath="/"
	src *dagger.Directory,
) *dagger.Container {
	pipCache := dag.CacheVolume("pip-cache")
	opts := dagger.ContainerWithDirectoryOpts{
		Exclude: []string{
			".dagger/",
			".git",
			".gitignore",
			".github",
			".dvc",
			"LICENSE",
			"dagger.json",
			"notebooks/",
			"models/*",
			".ruff_cache/",
			".pytest_cache/",
			"data/**/*.csv",
			"mlruns/",
		},
	}
	return dag.
		Container().
		From("python:3.12.2-bookworm").
		WithDirectory("/app", src, opts).
		WithWorkdir("/app").
		WithMountedCache("/root/.cache/pip", pipCache).
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithExec([]string{"pip", "install", "dvc"}).
		WithExec([]string{"dvc", "get", "https://github.com/Jeppe-T-K/itu-sdse-project-data", "raw_data.csv", "-o", "data/raw"})
}
