# Data Version Control with DVC

## Installation

`dvc` is already part of the lab environment (`lab06/env.yaml`), so there is nothing extra to install:

```shell
conda activate mlops-lab-06
dvc version   # should print 3.67.1
```

If you prefer installing it in a different manner, please refer to [the dvc documentation](https://dvc.org/doc/install/).

> DVC is a separate tool that works alongside Git - it is **not** built on Git LFS and does not need it. Only local and a few other remotes work out of the box; for cloud storage you install the matching extra, e.g. `pip install "dvc[s3]"` or `pip install "dvc[gdrive]"`.

## Data Management with DVC

DVC lets you capture the versions of your data (and models) in Git commits, while storing the actual data outside of the repository.
It also provides you with a mechanism to switch between the different data contents. When used regularly and diligently, the result is a single history of data, code, and ML models.

DVC uses what they refer to as _versioning through codification_. You produce _metafiles_ once, which describe what datasets, ML artifacts, etc. to track. This metadata can be put in Git in place of large files.
Once that's done, you can use DVC to create snapshots of the data, restore previous version, reproduce experiments - and much more.

There are _many_ features and use cases for DVC. We focus on the most basic one - versioning data and sharing it through a remote - and let you explore the remaining ones [in the DVC docs](https://dvc.org/doc/use-cases) (e.g. a _data registry_, a central repository for all your datasets, or DVC pipelines).

We will start with the basics of DVC, then you will put a dataset under DVC control yourself. In the augmentation notebook, you will then change the data and version the changes.

### DVC 101

DVC is modelled after, and built on top of, git, and a _DVC project_ always goes together with a git repository.

To create a new DVC project, you first have to create a git repository (i.e. use `git init`).
After initializing the git repository, you can initialize the DVC project. You do this using [`dvc init`](https://dvc.org/doc/command-reference/init).

```shell
dvc init
```

Upon running `dvc init`, a [directory `.dvc`](https://dvc.org/doc/user-guide/project-structure/internal-files) is created (similar to how `git init` creates `.git`). `.dvc` creates configuration files and a cache for project data. The details are not important for our purposes.

Should you be unhappy with DVC, you can use [`dvc destroy`](https://dvc.org/doc/command-reference/destroy) to remove all DVC-specific files from the directory. This is synonymous with "deleting" the DVC project.

Suppose you have a big file, e.g. `hour.csv`. Because it is so big, we cannot check it into Git directly. To track this file with DVC instead, we can use `dvc add`:

```shell
dvc add hour.csv
```

This will do the following:

1. The file content is moved into the cache (`.dvc/cache`), and the file in your workspace is linked (or copied) back from there - so `hour.csv` is still where it was.
2. A small pointer file `hour.csv.dvc` is created next to it. It contains the hash of the file, and it is what you commit to Git instead of the data.
3. `hour.csv` is added to `.gitignore` to prevent Git from tracking the file itself.

In the end, the situation looks as depicted in the image below:

![alt text](imgs/dvc_versioning.png)
(Image taken from [here](https://dagshub.com/blog/getting-started-with-dvc/).)

What happens if you modify a file tracked by DVC? When you change `hour.csv` and later add this change to DVC (`dvc add hour.csv` again, or `dvc commit`), DVC
will copy the new version to the cache and update the hash in the pointer. After committing the pointer to Git, every Git commit refers to exactly one version of the data.

#### Remotes

The description above might have you left wondering about how you can share the different file versions with your collaborators.
After all, the cache is local. This is where remotes come in. In their own words,

> DVC remotes provide access to external storage locations to track and share your data and ML models. Usually, those will be shared between devices or team members who are working on a project. For example, you can download data artifacts created by colleagues without spending time and resources to regenerate them locally.

There are two main uses of remote storage:

1. synchronization of large files and directories tracked by DVC
2. centralization of data storage for sharing and collaboration

DVC supports a range of different providers and storage types such as a local or network directory, `SSH`, `S3`, `Google Drive`, [and more](https://dvc.org/doc/user-guide/data-management/remote-storage).

#### Adding and tracking data

There are many ways of adding data to your DVC project. Here are a few:

First off, there's `dvc add`, which you've already learnt about above. This tracks data files or directories with DVC. If you think back to the previous part
on `git lfs`, this step corresponds to `git lfs track` followed by `git add`.

Then, there are commands to add external data to your DVC project. These commands come in two flavours, `get` and `import`.
`get` commands do **not** track the downloaded data files, while `import` commands **do** track the files.

Then, there are also `-url` and non `-url` commands. The commands with a `-url` suffix can be used to add files that are external to DVC or git (i.e., they are not already tracked by DVC or Git), while the commands without a suffix can only be used with files that are tracked in another DVC or Git repository:

- `dvc get`: Download a file or directory tracked by DVC or by Git into the _current working directory_. This command does **not** track the downloaded files.
- `dvc get-url`: Download a file or directory from a supported URL, for example `s3://`, `ssh://`, and other protocols, into the local file system. This is comparable to a command like `wget` or `curl`.
- `dvc import`: Download a file or directory tracked by another DVC or Git repository into the workspace, and track it (an import `.dvc` file is created).
- `dvc import-url`: Download a file or directory from a supported URL, for example `s3://`, `ssh://`, and other protocols, and track it.

A small comment on tracking. When you track data (`dvc add`, `dvc import`), DVC will compute hashes which it later uses to check whether modifications have been made. If you have a lot of data in your repository, this will eventually become very expensive. So, instead of tracking all the files individually, you might be tempted to instead track e.g. a ZIP archive or a tarball of the data. Note, however, that you will lose the ability to efficiently track changes that were made to your data if you end up using tarballs, because the tarball hash will be different every time - you might as well use LFS in this case. (There are still cases where you maybe want to `add` tarballs to DVC - more on this later.) Don't forget that you should only commit changes to the data that you want to keep, e.g. large augmentations or newly added samples, so you won't run `dvc add` very frequently.

#### A note on "workspaces"

If you read the DVC documentation, you will come across the concept of a _DVC workspace_. This is again something that
DVC borrows from Git. In Git,

> A repository can have zero (i.e. bare repository) or one or more worktrees attached to it. One "worktree" consists of a "working tree" [= the tree of actual checked out files] and repository metadata, most of which are shared among other worktrees of a single repository, and some of which are maintained separately per worktree (e.g. the index, HEAD and pseudorefs like MERGE_HEAD, per-worktree refs and per-worktree configuration file).

A DVC workspace is analogous to a Git working tree. It is simply the currently visible version of the project.

## DVC hands-on

Enough theory, let's get started. First, create a new, empty repository on GitHub (not the one you used for Git LFS) and clone it.
Then, in the root of your repository, run

```shell
dvc init
```

To see the files that DVC created, run `git status`. The output should look something like the following:

`dvc init` already stages the new files for you (`git add`):

```raw
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   .dvc/.gitignore
        new file:   .dvc/config
        new file:   .dvcignore
```

Commit these files to git.

```shell
git commit -m "Initialize DVC"
```

Now we are ready to add data to our DVC repository. Create the `data` directory and change into it (only for the download - we will go back to the repository root afterwards).

```shell
mkdir data
cd data
```

We'll again use the `102flowers` dataset. Download the tarball using DVC but don't track it.
The download URL is

```raw
https://thor.robots.ox.ac.uk/flowers/102/102flowers.tgz
```

<details>
    <summary>Solution</summary>

```shell
dvc get-url <URL>
```

</details>

Once that's completed, extract the tarball

```shell
# Extract the contents of the tarball.
tar -xvf 102flowers.tgz

# Rename the output folder
mv jpg 102flowers

# Delete the tarball
rm 102flowers.tgz

# Go back to the repository root
cd ..
```

You should now have a folder `data/102flowers` with 8189 images.

Now, track this folder with DVC! (Computing the hashes of all 8189 files takes a moment.)

<details>
    <summary>Solution</summary>

```shell
dvc add data/102flowers
```

</details>

Let's inspect the resulting "pointer":

```shell
cat data/102flowers.dvc
```

```raw
outs:
- md5: 56eae55ec88caaf5b57e6ae9314484d4.dir
  size: 346809251
  nfiles: 8189
  hash: md5
  path: 102flowers
```

As you can see, there's the directory hash (`md5`),
the `size` and number of files `nfiles` along with the directory path (relative to the `.dvc` file). Your hash may differ if the dataset has changed.

Let's commit the pointer and `.gitignore` to git and push it to the remote.

<details>
    <summary>Solution</summary>

```shell
git add data/.gitignore data/102flowers.dvc
git commit -m "Add 102flowers"
git push
```

</details>

Last, but definitely not least, let's push the data to a remote location. As we've mentioned above, DVC supports many
storage locations. For the sake of simplicity, we will use a local path.

Create a directory somewhere on your system, _outside_ of your repository. We opt for the temporary filesystem `/tmp`:

```shell
mkdir /tmp/dvcdata
```

(`/tmp` is emptied when the machine restarts, and on a shared machine other users may use the same name - pick a unique name such as `/tmp/dvcdata-$USER` there. For a real project, the remote would be an S3 bucket, a network drive, ... that your whole team can access.)

Adding a remote in DVC follows the same syntax as Git:

```shell
dvc remote add <name> <url>
```

where url can also be a local path.
Add the temporary directory you create before as a remote.
Make sure to use the _absolute_ path (`/very/long/path`) and not a relative one (`./my/path` or `../my/path`).
Relative paths work, but may lead to surprises.

_Hint_: You can designate a remote the _default remote_ by adding `-d` or `--default` to the command.

<details>
    <summary>Solution</summary>

```shell
dvc remote add -d temporary /tmp/dvcdata
```

</details>

This will add an entry to `.dvc/config`, which you have to commit to your git repository to
persist.

```shell
git add .dvc/config
git commit -m "Add remote to config"
git push
```

To push your data to the remote, simply run

```shell
dvc push
```

Now, take a leap of faith and delete your local copy of the repository and
clone it again from GitHub. Note how there is no data in `data/` - only `102flowers.dvc` and `.gitignore`!
To "download" the data from the remote, run (in the new clone)

```shell
dvc pull
```

This works because the remote configuration is stored in `.dvc/config`, which you committed to Git. (It only works on the same machine, though: the remote is a local directory. A collaborator on another machine would need access to the same storage.)

---

That's it, you've mastered the basics of DVC! Don't delete this repository just yet, we will
use it again in the [augmentation notebook](./notebooks/albumentations.ipynb).
