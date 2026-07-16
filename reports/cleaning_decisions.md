# Data Cleaning Decisions

## Decision 1: Joint Application Columns

Columns:
- annual_income_joint
- verification_income_joint
- debt_to_income_joint

Observation:
These columns have approximately 85% missing values.

Investigation:
The dataset contains:
- 8505 individual applications
- 1495 joint applications

Conclusion:
The missing values are expected because individual loan applications do not have joint applicant information.

Decision:
Retain these columns for now. They may be useful when analyzing only joint applications.

## Decision: num_accounts_120d_past_due

Missing Values: 318 (3.18%)

Observation:
The only observed value in the dataset was 0.

Decision:
Filled missing values with 0.

Reason:
This column represents a count of accounts more than 120 days past due. Since all available records contained 0 and no positive counts were present, filling missing values with 0 was considered a reasonable assumption for this project. This assumption should be revisited if additional metadata becomes available.