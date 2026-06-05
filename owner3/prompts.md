# Prompt Contract

## System Prompt

```text
You are a careful radiologist. Answer the question precisely in the exact format requested.
```

## Binary Question

```text
Based on this chest X-ray, is there radiographic evidence of pneumonia? Answer exactly PRESENT or ABSENT.
```

## Abstention Question

```text
Based on this chest X-ray, is there radiographic evidence of pneumonia? If the image and the lab value point in different directions, you may answer CONFLICT. Answer exactly PRESENT, ABSENT, or CONFLICT.
```

## Lab Statement

```text
The patient's white blood cell count is {value} K/uL (reference {lo}-{hi}), {qual}.
```

Where `qual` is:

- `within normal limits`
- `which is elevated`
- `which is low`

## Clinical-Lean Sentence

High WBC:

```text
An elevated white blood cell count raises concern for an active infection such as pneumonia.
```

Normal/low WBC:

```text
A white blood cell count that is not elevated makes an active bacterial infection less likely.
```
