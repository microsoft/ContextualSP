# ProPara Evaluator

## Example

```
% export PYTHONPATH=.
% python3 evaluator.py --predictions testfiles-1/predictions.tsv --answers testfiles-1/answers.tsv --output /tmp/metrics.json
=================================================
Question     Avg. Precision  Avg. Recall  Avg. F1
-------------------------------------------------
Inputs                0.793        0.597    0.681
Outputs               0.739        0.593    0.658
Conversions           0.878        0.200    0.326
Moves                 0.563        0.331    0.417
-------------------------------------------------
Overall Precision 0.743                          
Overall Recall    0.430                          
Overall F1        0.545                          
=================================================

Evaluated 54 predictions against 54 answers.
% cat /tmp/metrics.json
{"precision": 0.743, "recall": 0.43, "f1": 0.545}
```

## Usage

The script requires prediction and answer input files, and produces a report to
standard out. You'll need Python 3.6 or newer to run it.

Optional:

* the argument `--output` writes the overall precision, recall and F1 score to a JSON file.
* the argument `--diagnostics` writes a diagnostic file with process summaries (an intermediate representation of each process) and their scores.
* the argument `--sentences` reads sentences from a sentences file and includes them in the diagnostics file. See [sentences.tsv files in the ../data directory](../data/).

## Evaluation process

### Overview

The task is to predict what happens to specific participants in each sentence
of a process paragraph. This prediction for a process is summarized to answer
four key questions:

1. **What are the Inputs?** (i.e., Which participants existed before the process began, and don't exist after the process ended? Or, what participants were consumed?)
2. **What are the Outputs?** (i.e., Which participants existed after the process ended, but didn't exist before the process began? Or, what participants were produced?)
3. **What are the Conversions?** (i.e., Which participants were converted to which other participants?)
4. **What are the Moves?** (i.e., Which participants moved from one location to another?)

The proposed answers (prediction) to these questions are compared to the
correct answers (gold) to arrive at a score for each question.

These scores are aggregated over all processes to arrive at a final performance
score represented by precision, recall and F1 calculations.

### Details

The process of evaluating predictions can be divided into four steps.

#### Step 1. Reading the action file

An **action file** is a file with tab-separated values representing a table of
actions for participants in sentences of process paragraphs.

For example, consider process 1167 from the training set, consisting of five
sentences:

> ① The gravity of the sun pulls its mass inward. ② There is a lot of pressure
> on the Sun. ③ The pressure forces atoms of hydrogen to fuse together in
> nuclear reactions. ④ The energy from the reactions gives off different kinds
> of light. ⑤ The light travels to the Earth.

For this process paragraph, the objective is to decide what happens to the two
participants "atoms of hydrogen" and "sunlight; light" by filling in the three
blank columns in a six-column table:

| ProcessID | Sentence | Participant(s)    | Action  | Location Before   | Location After |
| --------- | -------- | ----------------- | ------  | ----------------- | -------------- |
| 1167      | 1        | atoms of hydrogen |         |                   |                |
| 1167      | 1        | sunlight; light   |         |                   |                |
| 1167      | 2        | atoms of hydrogen |         |                   |                |
| 1167      | 2        | sunlight; light   |         |                   |                |
| 1167      | 3        | atoms of hydrogen |         |                   |                |
| 1167      | 3        | sunlight; light   |         |                   |                |
| 1167      | 4        | atoms of hydrogen |         |                   |                |
| 1167      | 4        | sunlight; light   |         |                   |                |
| 1167      | 5        | atoms of hydrogen |         |                   |                |
| 1167      | 5        | sunlight; light   |         |                   |                |

A TSV file with this table is called an **action file** because each row in the
table represents what happens to a specific participant in a specific sentence
in a specific process paragraph.

The first three columns are fixed, and provided to you:

* **ProcessID** (column 1): This is the identifier for the process paragraph.
* **Sentence** (column 2): This is the sentence number (starting with 1) in the process paragraph.
* **Participant(s)** (column 3): This is usually a span from the process paragraph identifying an interesting participant. It may contain a `;` character to delimit alternative participants (e.g., coreferent mentions of `sunlight` and `light`.) See section **Scoring each process** for another example.

Your prediction action file must contain these three columns verbatim to be admitted for evaluation.

The last three columns are to be predicted.

* **Action** (column 4): This should describe what happens to the participant in this sentence. It must be one of `NONE`, `CREATE`, `MOVE`, or `DESTROY`.
* **Location Before** (column 5): This should describe the location of the participant (column 3) before this sentence (column 2).
* **Location After** (column 6): This should describe the location of the participant (column 3) after this sentence (column 2).

Specifically, there are rules for the locations based on the kind of action happening.

* If the **Action** is `NONE`, then...
  * **Location Before** and **Location After** must be equal. The special value `?` means the participant's location is unknown to the predictor, and the special value `-` means the participant doesn't exist.

* If the **Action** is `CREATE`, then...
  * **Location Before** must be `-` to mean that the participant didn't exist before this sentence.
  * **Location After** is the location where the participant was created. The special value `?` means the participant was created but the location is unknown to the predictor. 

* If the **Action** is `MOVE`, then...
  * **Location Before** is the location from where the participant moved. The special value `?` means the participant existed before this sentence, but its location is unknown to the predictor.
  * **Location After** is the location to where the participant moved. The special value `?` means the participant existed after this sentence, but its location is unknown to the predictor.

* If the **Action** is `DESTROY`, then...
  * **Location Before** is the location where the participant was destroyed. The special value `?` means the participant was destroyed but the location is unknown to the predictor.
  * **Location After** must be `-` to mean that the participant didn't exist after this sentence.
  
If your prediction file does not meet these requirements, evaluation will abort.

For example, a valid prediction for the above paragraph might be:

| ProcessID | Sentence | Participant(s)    | Action  | Location Before   | Location After |
| --------- | -------- | ----------------- | ------  | ----------------- | -------------- |
| 1167      | 1        | atoms of hydrogen | NONE    | -                 | -              |
| 1167      | 1        | sunlight; light   | NONE    | -                 | -              |
| 1167      | 2        | atoms of hydrogen | NONE    | -                 | -              |
| 1167      | 2        | sunlight; light   | NONE    | -                 | -              |
| 1167      | 3        | atoms of hydrogen | DESTROY | star              | -              |
| 1167      | 3        | sunlight; light   | NONE    | -                 | -              |
| 1167      | 4        | atoms of hydrogen | NONE    | -                 | -              |
| 1167      | 4        | sunlight; light   | CREATE  | -                 | star           |
| 1167      | 5        | atoms of hydrogen | NONE    | -                 | -              |
| 1167      | 5        | sunlight; light   | MOVE    | star              | soil           |

This means:

* In paragraph 1167, sentence 3, the participant `atoms of hydrogen` is destroyed at `star`
* In paragraph 1167, sentence 4, the participant `sunlight; light` is created at `star`
* In paragraph 1167, sentence 5, the participant `sunlight; light` is moved from `star` to `soil`

**Note:** This is a somewhat contrived example to illustrate the scoring mechanism below.

For comparison, the action file with the correct ("gold") answers is:

| ProcessID | Sentence | Participant(s)    | Action  | Location Before   | Location After |
| --------- | -------- | ----------------- | ------  | ----------------- | -------------- |
| 1167      | 1        | atoms of hydrogen | NONE    | sun               | sun            |
| 1167      | 1        | sunlight; light   | NONE    | sun               | sun            |
| 1167      | 2        | atoms of hydrogen | DESTROY | sun               | -              |
| 1167      | 2        | sunlight; light   | NONE    | -                 | -              |
| 1167      | 3        | atoms of hydrogen | NONE    | -                 | -              |
| 1167      | 3        | sunlight; light   | NONE    | -                 | -              |
| 1167      | 4        | atoms of hydrogen | NONE    | -                 | -              |
| 1167      | 4        | sunlight; light   | CREATE  | -                 | sun            |
| 1167      | 5        | atoms of hydrogen | NONE    | -                 | -              |
| 1167      | 5        | sunlight; light   | MOVE    | sun               | earth          |

That is:

* In paragraph 1167, sentence 3, the participant `atoms of hydrogen` is destroyed at `sun`
* In paragraph 1167, sentence 4, the participant `sunlight; light` is created at `sun`
* In paragraph 1167, sentence 5, the participant `sunlight; light` is moved from `sun` to `earth`

**Note:** You can use the `explainer.py` program to parse action files into explanations like the above.

Next, these two action files are summarized.

#### Step 2. Summarizing each process

To compare predicted actions to answer actions, each process paragraph in these
tables is summarized into answers to the four questions described above.

For the above predictions, the internal summary can be seen in the diagnostics
output like this:

```json
{
  "prediction_summary": {
    "process_id": 1167,
    "inputs": { "participants": [ "atoms of hydrogen" ] },
    "outputs": { "participants": [ "sunlight OR light" ] },
    "conversions": null,
    "moves": [
      {
        "participants": "sunlight OR light", "step_number": 5,
        "location_before": "star", "location_after": "soil"
      } 
    ] 
  } 
}   
```

For the corresponding answer, the internal summary in the diagnostics output looks like this:

```json
{
  "answer_summary": {
    "process_id": 1167,
    "inputs": { "participants": [ "atoms of hydrogen" ] },
    "outputs": { "participants": [ "sunlight OR light" ] },
    "conversions": [
      {
        "participants_destroyed": "atoms of hydrogen",
        "participants_created": "sunlight OR light",
        "location": "sun", "step_number": 3
      }
    ],
    "moves": [
      {
        "participants": "sunlight OR light", "step_number": 5,
        "location_before": "sun", "location_after": "earth"
      }
    ]
  }
}
```

To read the code that summarizes process paragraphs from an action file, look
at the function `summarize_actions_file` and its uses.

#### Step 3. Scoring each process

The summary of a prediction can be compared to the corresponding answer by
assigning precision and recall scores to each of these four questions.
Internally, this is represented in JSON like this:

```json
{
  "score": {
    "process_id": 1167,
    "inputs": { "precision": 1, "recall": 1 },
    "outputs": { "precision": 1, "recall": 1 },
    "conversions": { "precision": 1, "recall": 0 },
    "moves": { "precision": 0.3333333333333333, "recall": 0.3333333333333333 }
  }
}
```

In this case, the precision and recall of **What are the Inputs?** and **What
are the Outputs?** is 1, because the prediction and answer both have the same
summarization, even though the predicted locations of `CREATE` and `DESTROY`
actions differ.

Since the prediction did not describe any conversions, but the answers do, the
recall for **What are the Conversions?** is 0.

Finally, while the prediction did correctly describe that the participant
`sunlight OR light` moved in sentence 5, the before and after locations are not
correct. Therefore, the precision and recall suffer for the question **What are
the Moves?**

To read the code that compares these summaries, look at the functions
`_score_inputs`, `_score_outputs`, `_score_conversions`, and `_score_moves` and
their use.

**Note about location comparison:**

> Locations are normalized before comparison.
> 
> Since you have to discover locations in the paragraph yourself when predicting
> actions for specific participants, your locations may differ slightly from the
> ones in the answers.
> 
> For example, if the paragraph has `Ants walk on branches` and `The ant fell off
> a branch` then the location of the ant could be written as `branches`, `a
> branch`, or `branch`. Since your chosen span of the paragraph for this location
> may differ from the correct answer, a process of normalization resolves any of
> these variants to `branch`: first by lower casing the string, then removing
> starting articles like `a` and finally using the Porter stemming algorithm to
> settle on the final string.
> 
> To see this in the code, look for the function `_compare_locations` and its uses.

**Note about participant comparison:**

> Participants are not normalized before comparison.
>
> Your predictor may have selected a participant from the process paragraph that is
> not the one chosen for you in the answer. From the example above, if you predict
> an action on the participant `Ants`, but the answer action is on participant `ant`,
> your predicted action will not be matched.
>
> To see this in the code, look for the function `_compare_participants` and its uses.
> 
> If your prediction refers to participants that are not in the answers, you'll see a
> report alerting you to the difference and the evaluation will abort. To see this in
> the code, look for the function `diff_participants` and its use. You should correct
> these differences by predicting actions only on the participants chosen for you. That is,
> your prediction's first three columns should match the first three columns of the answer
> file.

#### Step 4. Calculating an overall score

The above process scores are aggregated to an overall performance score.

To illustrate, consider the precision and recall scores (as computed above), in a table:

<html>
<table>
  <tr>
    <th rowspan="2">ProcessID</th>
    <th colspan="4">Precision</th>
    <th colspan="4">Recall</th>
  </tr>
  <tr>
    <td>Inputs</td>
    <td>Outputs</td>
    <td>Conversions</td>
    <td>Moves</td>
    <td>Inputs</td>
    <td>Outputs</td>
    <td>Conversions</td>
    <td>Moves</td>
  </tr>
  <tr>
    <td>1</td>
    <td>IP<sub>1</sub></td>
    <td>OP<sub>1</sub></td>
    <td>CP<sub>1</sub></td>
    <td>MP<sub>1</sub></td>
    <td>IR<sub>1</sub></td>
    <td>OR<sub>1</sub></td>
    <td>CR<sub>1</sub></td>
    <td>MR<sub>1</sub></td>
  </tr>
  <tr>
    <td>2</td>
    <td>IP<sub>2</sub></td>
    <td>OP<sub>2</sub></td>
    <td>CP<sub>2</sub></td>
    <td>MP<sub>2</sub></td>
    <td>IR<sub>2</sub></td>
    <td>OR<sub>2</sub></td>
    <td>CR<sub>2</sub></td>
    <td>MR<sub>2</sub></td>
  </tr>
  <tr>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
</table>
</html>

Given these precision and recall values, the overall performance is calculated as three numbers:

* **Overall Precision P** = (Average of all IP + Average of all OP + Average of all CP + Average of all MP) / 4
* **Overall Recall R** = (Average of all IR + Average of all OR + Average of all CR + Average of all MR) / 4
* **Overall F1 score** = harmonic mean of P and R = 2 * (P * R) / (P + R)

To read the code that calculates these final scores, look at the class `Evaluation`.

## Evaluator Development

### Testing

Run this script to run a comprehensive suite of tests:

```bash
./test.sh
```
