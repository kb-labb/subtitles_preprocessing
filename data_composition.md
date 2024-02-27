# Datasammanställning

| Data           | Casing      | Punctuation    | Size [GB]      | Size [GB] (after dedup) | Size [GB] (after lang=se) |  Duration [h] | Duration [h] (after dedup) | Duration [h] (after lang=se) | format   |
| -------------- | ----------- |--------------  | -------------- | ------------------------|-------------------------- | ------------- | -------------------------- | ---------------------------- | -------- |
| NST            | v           | v              | 96.5           |                         |                           | -             |                            |                              | HF       |
| CommonVoice 16 | v           | v              | 1.14           |                         |                           | 46            |                            |                              | HF       |
| FLEURS         | v           | v              | -              |                         |                           | 12            |                            |                              | HF       |
| SVT            | v           | v              | 300            |                         |                           | 1500          |                            |                              | pipeline |
| smdb01         | v           | v              | 5.3 (sub only) |                         |         105               | 44 000        |          8475 / 2          |          752                 | pipeline |
| smdb04         | v           | v              | 13 (sub only)  |                         |          97               | 126 000       |          8475 / 2          |          672                 | pipeline |
| youtube        | ?           | ?              |                |                         |                           | 8110          |  no dedup                  | 6033 (bleu > 0.3)            | pipeline |
| Riksdagsanf.   | ?           | ?              | ?              |                         |                           | 5000          |                            |                              | pipeline |
| Swedia         | v           | v              | ?              |                         |                           | 144           |                            |                              | pipeline |
