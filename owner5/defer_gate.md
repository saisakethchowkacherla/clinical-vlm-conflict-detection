# Defer Gate

## Goal

Detect when image-only and image+lab answers disagree.

## Rule

If image_only_answer != image_lab_answer

Output = CONFLICT

Else

Output = image_lab_answer
