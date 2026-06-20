# Component and Runtime Summary

## Demo100 Main Table

| Method | Dataset | BLEU-4 | ROUGE-L | CIDEr | Avg Lat (s) | P95 Lat (s) | Avg Prompt Tok | Avg Total Tok | Success |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chrono-full | CRONQUESTION | 0.2715 | 0.4878 | 2.5643 | 2.216 | 4.726 | 614.45 | 630.71 | 1.00 |
| Chrono-no_memory | CRONQUESTION | 0.1785 | 0.3396 | 1.6276 | 1.874 | 3.013 | 323.90 | 340.83 | 1.00 |
| Chrono-relation_only | CRONQUESTION | 0.2963 | 0.4783 | 2.6533 | 2.045 | 3.611 | 620.68 | 637.01 | 1.00 |
| KQG-CoT | CRONQUESTION | 0.1904 | 0.4112 | 1.9347 |  |  |  |  |  |
| Chrono-full | MULTITQ | 0.3087 | 0.5266 | 2.7035 | 1.956 | 3.338 | 690.90 | 708.43 | 1.00 |
| Chrono-no_memory | MULTITQ | 0.1479 | 0.3910 | 1.3809 | 4.201 | 4.028 | 342.22 | 358.74 | 1.00 |
| Chrono-relation_only | MULTITQ | 0.2635 | 0.4839 | 2.2389 | 1.996 | 3.299 | 685.38 | 702.62 | 1.00 |
| KQG-CoT | MULTITQ | 0.2186 | 0.4523 | 1.8383 |  |  |  |  |  |

## Full Test Main Table

| Method | Dataset | N | BLEU-4 | ROUGE-L | CIDEr | Avg Lat (s) | P95 Lat (s) | Avg Total Tok | Success |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chrono-full | CRONQUESTION | 19122 | 0.2947 | 0.5059 | 2.7356 | 1.967 | 3.529 | 636.14 | 1.00 |
| Chrono-no_memory | CRONQUESTION | 19122 | 0.1890 | 0.3538 | 1.6444 | 1.713 | 2.304 | 343.91 | 1.00 |
| KQG-CoT | CRONQUESTION | 19122 | 0.2124 | 0.4268 | 2.0148 |  |  |  |  |
| Chrono-full | MULTITQ | 36318 | 0.3042 | 0.5227 | 2.6172 | 1.790 | 2.582 | 699.20 | 1.00 |
| Chrono-no_memory | MULTITQ | 36318 | 0.1449 | 0.3939 | 1.2250 | 4.302 | 8.589 | 359.07 | 1.00 |
| KQG-CoT | MULTITQ | 36318 | 0.1810 | 0.4286 | 1.3367 |  |  |  |  |

## Component Trade-off Demo100

| Method | Dataset | BLEU-4 | CIDEr | Avg Lat (s) | Avg Total Tok |
|---|---|---:|---:|---:|---:|
| Chrono-full | CRONQUESTION | 0.2715 | 2.5643 | 2.216 | 630.71 |
| Chrono-no_memory | CRONQUESTION | 0.1785 | 1.6276 | 1.874 | 340.83 |
| Chrono-relation_only | CRONQUESTION | 0.2963 | 2.6533 | 2.045 | 637.01 |
| Chrono-full | MULTITQ | 0.3087 | 2.7035 | 1.956 | 708.43 |
| Chrono-no_memory | MULTITQ | 0.1479 | 1.3809 | 4.201 | 358.74 |
| Chrono-relation_only | MULTITQ | 0.2635 | 2.2389 | 1.996 | 702.62 |

## Component Trade-off Full Test

| Method | Dataset | BLEU-4 | CIDEr | Avg Lat (s) | Avg Total Tok |
|---|---|---:|---:|---:|---:|
| Chrono-full | CRONQUESTION | 0.2947 | 2.7356 | 1.967 | 636.14 |
| Chrono-no_memory | CRONQUESTION | 0.1890 | 1.6444 | 1.713 | 343.91 |
| Chrono-full | MULTITQ | 0.3042 | 2.6172 | 1.790 | 699.20 |
| Chrono-no_memory | MULTITQ | 0.1449 | 1.2250 | 4.302 | 359.07 |

## Scalability (MULTITQ, Chrono-full)

| Train Scale | BLEU-4 | ROUGE-L | CIDEr | Avg Lat (s) | P95 Lat (s) | Throughput (/min) | Avg Total Tok | Cache Build (s) | Cache Size (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.3380 | 0.5658 | 3.1234 | 6.181 | 8.881 | 240.293 | 701.78 | 15.126 | 15.43 |
| 40 | 0.3204 | 0.5519 | 2.9846 | 7.433 | 11.377 | 196.790 | 703.45 | 30.582 | 22.12 |
| 80 | 0.3388 | 0.5641 | 3.1753 | 6.253 | 10.772 | 261.993 | 702.86 | 61.102 | 31.31 |
| 100 | 0.3242 | 0.5419 | 2.8817 | 7.390 | 10.652 | 194.689 | 708.61 | 74.572 | 35.06 |
