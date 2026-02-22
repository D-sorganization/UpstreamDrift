# GitHub Issue Draft: Data Copyright Risk in Validation Data

**Title:** [Legal] Data Copyright Risk in Validation Data
**Labels:** legal, copyright, data-compliance

## Description
The validation module `src/shared/python/validation_pkg/validation_data.py` contains a hardcoded dataset `PGA_TOUR_2024` which is explicitly attributed to "trackman.com".

### Findings
- The data structure matches a proprietary dataset (PGA Tour Averages).
- Source is listed as "trackman.com".
- While factual data (averages) is generally not copyrightable, the extraction and redistribution of a database or collection from a competitor's website may violate Terms of Service or Database Rights (in certain jurisdictions).

## Risks
- **Legal/Copyright:** Potential infringement of Database Rights or Breach of Contract (ToS) if scraped.
- **Reputation:** Using competitor data without clear license.

## Proposed Action
1. **Verify Source:** Determine if this data is from a public press release, open article, or a gated customer-only portal.
2. **Fair Use Assessment:** If used for educational/validation purposes only (and not resold), it may be fair use.
3. **Externalize:** Consider moving this data to an external config file that the user must provide, rather than shipping it in the source code.
4. **Attribution:** Ensure attribution is compliant with TrackMan's citation guidelines.

## Acceptance Criteria
- [ ] Legal review of "trackman.com" Terms of Use regarding data usage.
- [ ] Decision made: Keep (with citation), Remove, or Externalize.
- [ ] If kept, ensure comments explicitly state the "Fair Use" justification.
