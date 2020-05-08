---
title: Git Quick Guide
author: Sophie Marchand
date: May 2020
---

# First step
Open a terminal. Note that if you are on Window you can use [Git for Window](https://gitforwindows.org/). 

# Clone locally your repository
Create on your computer a git folder and go into it
```python
mkdir git
cd git
```

Get the ssh link from GitLab/GitHub project. Then, clone the repository on your computer
```python
git clone git@github.com:UserName/my_repository.git
```

# Create a new branch and/or switch branch
Go into the project folder
```python
cd my_repository
```

To create a new branch such that you can experiment modifications without others to be affected
```python
git checkout -b name_of_your_new_branch
```

To switch to an existing branch
```python
git checkout existing_branch_name
```

To check in which branch you are
```python
git branch
```

To check all the existing branches
```python
git branch --all
```

# Make modifications and update your code
To update your local modifications to the online repository, first add them locally
```python
cd my_repository
git add .
```
\pagebreak
Or, if you want to only add the modifications of one specific file
```python
git add file_name.extension
```

Then commit the added modification with a description
```python
git commit -m "description_modification"
```

Finally push them! For the first time you will need to do
```python
git push --set-upstream origin name_of_your_new_branch
```

Otherwise, to push just do
```python
git push
```

# More

If someone made a modification on the remote repository, you can get it locally by pulling
```python
git pull
```

To know if your branch need to be pushed or pulled, get its status
```python
git status
```

A nice graphical interface from git to follow the changes
```python
gitk
```
