# User Study Protocol Guide for OrbyGlasses

## What is a User Study and Why Do You Need It?

A **user study** is research where you test your technology with **real users** to:
1. **Validate it works**: Does OrbyGlasses actually help blind people navigate?
2. **Measure impact**: How much better is it than current methods (cane, guide dog)?
3. **Get feedback**: What features work? What needs improvement?
4. **Publish results**: Turn your project into a peer-reviewed research paper
5. **Attract funding**: Show investors/grants that you have real-world evidence

### Why This Makes Your Project Portfolio-Worthy

**Without user study**: "I built a cool AI navigation system" ✓
**With user study**: "I built AND scientifically validated a navigation system that reduces collisions by 47% in blind users" ✓✓✓

The difference is **proof of real-world impact** vs just a technical demo.

## What is IRB?

**IRB = Institutional Review Board**

An ethics committee that reviews research involving human participants to ensure:
- Participant safety
- Informed consent
- Privacy protection
- Ethical treatment

### Do You Need IRB Approval?

**You NEED IRB if**:
- You're affiliated with a university/institution
- You plan to publish in academic journals
- You're collecting sensitive health data
- Participants are vulnerable populations (blind users = yes)

**You might NOT need IRB if**:
- It's informal usability testing (not "research")
- You're just getting feedback (not measuring outcomes)
- No publication planned
- Very small scale (<5 people, no data collection)

**For OrbyGlasses**: If you want to publish or claim scientific validity, **GET IRB APPROVAL**

## User Study Design for OrbyGlasses

### Study Goal
**Primary Question**: Does OrbyGlasses improve navigation safety and independence for visually impaired users compared to traditional methods?

### Study Design: Randomized Controlled Trial (RCT)

#### Participants
- **Sample size**: 20-30 visually impaired adults
- **Inclusion criteria**:
  - Age 18-65
  - Legally blind or severely visually impaired
  - Can walk independently (with or without assistive device)
  - English speaking (for voice commands)

- **Exclusion criteria**:
  - Severe mobility impairments
  - Cognitive impairments affecting instruction following
  - Unable to wear glasses

#### Groups
1. **Experimental Group** (n=15): Use OrbyGlasses
2. **Control Group** (n=15): Use their usual method (white cane, guide dog, etc.)

#### Study Duration
- **Training period**: 1 week to learn OrbyGlasses
- **Testing period**: 4 weeks of regular use
- **Follow-up**: 2 weeks post-study interview

### Outcome Measures (What You Measure)

#### Primary Outcomes
1. **Collision Rate**
   - Number of collisions per hour of navigation
   - Measured by: Self-reported logs + accelerometer data (if IMU added later)

2. **Navigation Speed**
   - Time to complete standard route
   - Measured by: Stopwatch timing of indoor/outdoor courses

3. **User Confidence**
   - Self-reported confidence in navigation (1-10 scale)
   - Measured by: Weekly surveys

#### Secondary Outcomes
4. **Independence**
   - Hours per week navigating independently
   - Measured by: Weekly usage logs

5. **Quality of Life**
   - Standard QoL questionnaire (e.g., WHO-QOL)
   - Measured by: Pre/post surveys

6. **User Satisfaction**
   - System Usability Scale (SUS)
   - Measured by: Post-study survey

7. **Social Participation**
   - Number of independent trips per week (work, social, errands)
   - Measured by: Weekly logs

### Study Protocol (Step-by-Step)

#### Week 0: Recruitment & Baseline
1. **Recruit participants**
   - Contact local organizations (NFB, ACB, state services)
   - Post flyers at orientation & mobility centers
   - Social media outreach

2. **Screen participants**
   - Phone interview to check eligibility
   - Explain study purpose, risks, benefits

3. **Baseline assessment**
   - Consent form signing
   - Demographic survey
   - Baseline navigation test (timed route)
   - QoL and confidence surveys

#### Week 1: Training (Experimental Group Only)
1. **OrbyGlasses tutorial** (2 hours)
   - How to wear and operate
   - Understanding audio cues
   - Voice commands practice
   - Safety features

2. **Practice sessions** (3x 1-hour sessions)
   - Controlled environment (empty room)
   - Indoor navigation (hallways)
   - Outdoor navigation (sidewalks)

3. **Supervised trial**
   - Navigate familiar route with researcher present
   - Troubleshoot issues
   - Answer questions

#### Weeks 2-5: Testing Period
1. **Daily use**
   - Participants use OrbyGlasses for regular navigation
   - Keep usage logs (routes, duration, incidents)

2. **Weekly check-ins**
   - Phone call or in-person meeting
   - Collect logs
   - Administer surveys
   - Address technical issues

3. **Mid-study assessment (Week 3)**
   - Timed navigation test
   - Interim feedback survey

#### Week 6: Final Assessment
1. **Post-test navigation**
   - Same route as baseline
   - Compare times and collisions

2. **Final surveys**
   - QoL, confidence, satisfaction
   - Open-ended feedback

3. **Interview**
   - What worked well?
   - What needs improvement?
   - Would you continue using it?
   - Any safety concerns?

#### Weeks 7-8: Follow-up
1. **Retention check**
   - Are they still using OrbyGlasses?
   - Long-term satisfaction

### Data Collection Methods

#### Quantitative Data
```
Participant ID: ORB-001
Date: 2025-01-15
Condition: Experimental

Collision Count: 2 (baseline: 5)
Navigation Time: 8.5 min (baseline: 12.3 min)
Confidence Score: 8/10 (baseline: 4/10)
Usage Hours: 14 hrs this week
Independence Score: 9/10 (baseline: 5/10)
```

#### Qualitative Data
- Interview transcripts
- Open-ended survey responses
- Observational notes
- Technical issue logs

### Sample Consent Form (Simplified)

```
INFORMED CONSENT FORM

Study Title: Evaluation of OrbyGlasses AI Navigation System for Visually Impaired Users

Principal Investigator: [Your Name]

Purpose:
You are invited to participate in a research study testing a new navigation device
called OrbyGlasses. This study aims to evaluate whether OrbyGlasses improves
navigation safety and independence for people who are visually impaired.

Procedures:
If you agree to participate, you will:
1. Complete surveys about your navigation abilities and quality of life
2. Perform timed navigation tests
3. Use OrbyGlasses daily for 4 weeks (if assigned to experimental group)
4. Keep logs of your navigation activities
5. Attend weekly check-ins
6. Complete a final interview

Duration: 6 weeks total (1 training + 4 testing + 1 follow-up)

Risks:
- Possible discomfort wearing glasses
- Minimal risk of collision during navigation (no greater than daily activities)
- Frustration with technology learning curve
- Time commitment

Benefits:
- Potential improvement in navigation ability
- Contribution to assistive technology research
- Free use of OrbyGlasses during study
- Compensation: $200 for completion

Confidentiality:
Your data will be kept confidential. Only the research team will have access to
identifying information. Published results will use anonymous participant IDs.

Voluntary:
Your participation is completely voluntary. You may withdraw at any time without
penalty.

Contact:
Questions? Contact [Your Name] at [email] or [phone]
Concerns about rights? Contact [IRB office]

Consent:
I have read this form and agree to participate.

Signature: _________________ Date: _______
```

### IRB Application Checklist

To submit to IRB, you need:

1. **Protocol Document**
   - Study purpose and background
   - Methods and procedures
   - Risk/benefit analysis
   - Data management plan

2. **Consent Forms**
   - Written consent document
   - Verbal consent script (if applicable)

3. **Recruitment Materials**
   - Flyers
   - Social media posts
   - Email templates

4. **Surveys/Questionnaires**
   - All surveys participants will complete
   - Interview guides

5. **Data Security Plan**
   - How data will be stored (encrypted computer)
   - Who has access (only research team)
   - When data will be destroyed (e.g., 5 years post-publication)

6. **Investigator Qualifications**
   - Your CV/resume
   - CITI training certificate (required ethics training)

### Analysis Plan

#### Statistical Tests

**For Collision Rate:**
```python
# Compare experimental vs control
from scipy import stats

collision_experimental = [2, 1, 3, 0, 2, ...]  # n=15
collision_control = [5, 6, 4, 7, 5, ...]       # n=15

# Independent t-test
t_stat, p_value = stats.ttest_ind(collision_experimental, collision_control)

if p_value < 0.05:
    print("Significant reduction in collisions!")
```

**For Navigation Time:**
```python
# Paired t-test (before vs after for experimental group)
time_before = [12.3, 11.5, 13.2, ...]
time_after = [8.5, 7.8, 9.1, ...]

t_stat, p_value = stats.ttest_rel(time_before, time_after)
```

**For Confidence Scores:**
```python
# Wilcoxon signed-rank test (non-parametric for Likert scales)
confidence_before = [4, 3, 5, 4, ...]
confidence_after = [8, 7, 9, 8, ...]

stat, p_value = stats.wilcoxon(confidence_before, confidence_after)
```

#### Reporting Results

**Example Results Section:**
```
RESULTS

Participants:
30 visually impaired adults (mean age 42±12 years, 60% female)
- Experimental group (n=15): OrbyGlasses
- Control group (n=15): White cane (n=12) or guide dog (n=3)

Primary Outcomes:

1. Collision Rate:
   - Experimental: 1.8±0.9 collisions/hour (decreased from baseline 5.2±1.3)
   - Control: 5.1±1.5 collisions/hour (no change from baseline 5.0±1.4)
   - Difference: t(28)=7.2, p<0.001, Cohen's d=2.1 (large effect)
   → 65% reduction in collisions with OrbyGlasses

2. Navigation Speed:
   - Experimental: 8.3±1.2 min (improved from 12.1±1.8 min)
   - Control: 11.8±2.1 min (no change from 11.9±2.0 min)
   - Difference: t(28)=4.5, p<0.001
   → 31% faster navigation with OrbyGlasses

3. User Confidence:
   - Experimental: 8.2±1.1 (increased from 4.1±0.9)
   - Control: 4.3±1.2 (no change from 4.2±1.0)
   → 100% increase in confidence with OrbyGlasses

Qualitative Findings:
- 87% of participants reported OrbyGlasses "very helpful"
- Common praise: "Feels like having eyes again"
- Common issues: Battery life, voice recognition accuracy
```

## Publishing Your Results

### Target Venues

**Top Tier (Most Impact)**:
1. **CHI** (ACM Conference on Human Factors in Computing)
2. **ASSETS** (ACM Conference on Computers and Accessibility)
3. **UIST** (User Interface Software and Technology)

**Journals**:
1. **ACM TACCESS** (Transactions on Accessible Computing)
2. **IEEE TMI** (Transactions on Medical Imaging)
3. **Disability and Rehabilitation: Assistive Technology**

### Paper Structure

1. **Abstract** (250 words)
   - Problem, method, results, conclusion

2. **Introduction** (2-3 pages)
   - Why navigation is challenging for blind users
   - Limitations of current solutions
   - How OrbyGlasses addresses these

3. **Related Work** (2-3 pages)
   - Review of assistive navigation tech
   - Computer vision for accessibility
   - User studies in assistive tech

4. **System Design** (3-4 pages)
   - OrbyGlasses architecture
   - Technical implementation
   - User interface design

5. **User Study** (3-4 pages)
   - Methods (as described above)
   - Participants
   - Procedures
   - Measures

6. **Results** (3-4 pages)
   - Quantitative findings
   - Qualitative findings
   - Visualizations (graphs, quotes)

7. **Discussion** (2-3 pages)
   - What results mean
   - Implications for design
   - Limitations
   - Future work

8. **Conclusion** (1 page)
   - Summary of contributions

## Budget Estimate

| Item | Cost | Quantity | Total |
|------|------|----------|-------|
| Participant compensation ($200 each) | $200 | 30 | $6,000 |
| IRB application fee | $500 | 1 | $500 |
| OrbyGlasses prototypes | $300 | 5 | $1,500 |
| Survey tools (Qualtrics) | $50/mo | 3 | $150 |
| Data analysis software | $0 | - | $0 (use Python/R) |
| Publication fees | $1,000 | 1 | $1,000 |
| **TOTAL** | | | **$9,150** |

## Quick Start Without Full IRB

If you can't get IRB or funding yet, start with:

### Pilot Study (No IRB Needed)
1. **Usability testing** with 5-10 users
2. **Informal feedback** (not "research")
3. **No data publication** (just product improvement)
4. **Friends & family** testing first

### Then Scale Up
Once you have proof-of-concept:
1. Apply for small grants ($5K-$25K)
2. Partner with university for IRB
3. Run full study
4. Publish results

## Example Timeline

**Month 1**: IRB application + recruitment
**Month 2**: Participant screening + baseline
**Month 3**: Training + testing begins
**Month 4-5**: Ongoing testing
**Month 6**: Final assessments
**Month 7-8**: Data analysis
**Month 9-12**: Write paper + submit

**Total**: ~1 year from start to publication submission

---

## Bottom Line

A user study transforms OrbyGlasses from:
- "A cool project I built"
- TO: "A scientifically validated assistive technology with measurable impact"

This is the difference between a portfolio project and a **publication-worthy innovation**.

Even a small pilot study (5-10 users, no IRB) can provide powerful testimonials and evidence that this actually helps people. **Start small, scale up.**
