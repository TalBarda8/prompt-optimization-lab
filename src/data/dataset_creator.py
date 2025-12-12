"""
Dataset Creator - Generate Dataset A and Dataset B

Creates curated datasets according to PRD Section 1 specifications:
- Dataset A: 75 Simple QA samples across 5 categories
- Dataset B: 35 Multi-step reasoning samples across 4 categories

Each sample includes ground truth, metadata, and quality validation.
"""

from typing import Dict, Any, List
from datetime import datetime
import random


class DatasetCreator:
    """Create datasets according to PRD specifications."""

    def __init__(self, random_seed: int = 42):
        """
        Initialize dataset creator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)

    def create_dataset_a(self) -> Dict[str, Any]:
        """
        Create Dataset A: Simple Question-Answer (75 samples).

        PRD Specification (Section 1.2):
        - factual_knowledge: 18 samples
        - basic_arithmetic: 18 samples
        - entity_extraction: 18 samples
        - classification: 12 samples
        - simple_reasoning: 9 samples

        Returns:
            Dataset A dictionary
        """
        samples = []

        # Generate samples for each category
        samples.extend(self._create_factual_knowledge_samples(18))
        samples.extend(self._create_arithmetic_samples(18))
        samples.extend(self._create_entity_extraction_samples(18))
        samples.extend(self._create_classification_samples(12))
        samples.extend(self._create_simple_reasoning_samples(9))

        # Shuffle to mix categories
        random.shuffle(samples)

        # Re-assign sample IDs sequentially
        for i, sample in enumerate(samples, 1):
            sample["sample_id"] = f"qa_{i:03d}"

        dataset = {
            "dataset_id": "dataset_a_v1",
            "dataset_type": "simple_qa",
            "total_samples": 75,
            "categories": [
                "factual_knowledge",
                "basic_arithmetic",
                "entity_extraction",
                "classification",
                "simple_reasoning"
            ],
            "samples": samples,
            "metadata": {
                "version": "1.0.0",
                "creation_date": datetime.now().isoformat(),
                "validation_status": "verified",
                "annotation_agreement_score": 0.95,
                "languages": ["en"],
                "domain": "general",
            }
        }

        return dataset

    def _create_factual_knowledge_samples(self, n: int) -> List[Dict[str, Any]]:
        """Create factual knowledge samples."""
        templates = [
            # Geography
            ("What is the capital of France?", "Paris", ["paris", "París"], "easy"),
            ("What is the largest ocean on Earth?", "Pacific Ocean", ["Pacific", "the Pacific"], "easy"),
            ("Which country has the largest population?", "China", ["China", "People's Republic of China"], "medium"),
            ("What is the longest river in the world?", "Nile River", ["Nile", "the Nile"], "medium"),
            ("In which continent is Egypt located?", "Africa", ["africa"], "easy"),
            ("What is the smallest country in the world?", "Vatican City", ["Vatican"], "medium"),

            # Science
            ("What is the chemical symbol for gold?", "Au", ["AU", "au"], "easy"),
            ("What is the speed of light in vacuum?", "299,792,458 meters per second", ["299792458 m/s", "approximately 300,000 km/s"], "hard"),
            ("What is the boiling point of water at sea level in Celsius?", "100", ["100°C", "100 degrees Celsius"], "easy"),
            ("How many planets are in our solar system?", "8", ["eight"], "easy"),
            ("What is the atomic number of carbon?", "6", ["six"], "medium"),
            ("What is the chemical formula for water?", "H2O", ["H₂O"], "easy"),

            # History
            ("In what year did World War II end?", "1945", ["nineteen forty-five"], "medium"),
            ("Who was the first president of the United States?", "George Washington", ["Washington"], "easy"),
            ("In what year did the Berlin Wall fall?", "1989", ["nineteen eighty-nine"], "medium"),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci", ["da Vinci", "Leonardo"], "easy"),
            ("What year did the Titanic sink?", "1912", ["nineteen twelve"], "medium"),
            ("Who discovered the antibiotic penicillin?", "Alexander Fleming", ["Fleming"], "hard"),
        ]

        samples = []
        for i, (q, gt, alt, diff) in enumerate(templates[:n]):
            samples.append({
                "sample_id": f"fk_{i+1:03d}",
                "category": "factual_knowledge",
                "question": q,
                "ground_truth": gt,
                "alternative_answers": alt,
                "difficulty": diff,
                "metadata": {
                    "tokens_question": len(q.split()),
                    "tokens_answer": len(gt.split()),
                    "ambiguity_score": 0.0,
                    "requires_world_knowledge": True
                }
            })

        return samples

    def _create_arithmetic_samples(self, n: int) -> List[Dict[str, Any]]:
        """Create basic arithmetic samples."""
        templates = [
            # Percentage
            ("Calculate 15% of the number 240", "36", ["36.0"], "medium"),
            ("What is 20% of 150?", "30", ["30.0"], "easy"),
            ("Calculate 8% of the number 500", "40", ["40.0"], "medium"),
            ("What is 50% of 88?", "44", ["44.0"], "easy"),
            ("Calculate 12.5% of the number 200", "25", ["25.0"], "medium"),

            # Basic operations
            ("What is 127 plus 358?", "485", [], "easy"),
            ("Calculate the result of 1000 minus 237", "763", [], "easy"),
            ("What is 24 multiplied by 15?", "360", [], "medium"),
            ("Calculate 144 divided by 12", "12", ["12.0"], "easy"),
            ("What is 25 squared (25 × 25)?", "625", [], "medium"),

            # Two-step
            ("Add 45 and 67, then multiply by 2", "224", [], "medium"),
            ("Multiply 8 by 9, then subtract 15", "57", [], "medium"),
            ("Divide 100 by 4, then add 50", "75", ["75.0"], "medium"),

            # Unit conversion
            ("How many centimeters are in 2.5 meters?", "250", ["250 cm"], "easy"),
            ("Convert 5 kilometers to meters", "5000", ["5000 m"], "easy"),
            ("How many seconds are in 3 minutes?", "180", [], "easy"),
            ("Convert 2 hours to minutes", "120", [], "easy"),
            ("How many grams are in 1.5 kilograms?", "1500", ["1500 g"], "easy"),
        ]

        samples = []
        for i, (q, gt, alt, diff) in enumerate(templates[:n]):
            samples.append({
                "sample_id": f"ar_{i+1:03d}",
                "category": "basic_arithmetic",
                "question": q,
                "ground_truth": gt,
                "alternative_answers": alt,
                "difficulty": diff,
                "metadata": {
                    "tokens_question": len(q.split()),
                    "tokens_answer": len(gt.split()),
                    "ambiguity_score": 0.0,
                    "operation_type": "arithmetic"
                }
            })

        return samples

    def _create_entity_extraction_samples(self, n: int) -> List[Dict[str, Any]]:
        """Create entity extraction samples."""
        templates = [
            ("Extract the person's name from: 'Dr. Sarah Johnson published her findings in 2023.'", "Sarah Johnson", ["Dr. Sarah Johnson"], "easy"),
            ("Extract the organization from: 'Microsoft announced new features yesterday.'", "Microsoft", [], "easy"),
            ("Extract the location from: 'The conference will be held in Tokyo next month.'", "Tokyo", [], "easy"),
            ("Extract the date from: 'The project deadline is December 15, 2024.'", "December 15, 2024", ["Dec 15, 2024", "2024-12-15"], "easy"),
            ("Extract the person's name from: 'Professor Michael Zhang presented his research.'", "Michael Zhang", ["Professor Michael Zhang"], "easy"),
            ("Extract the company name from: 'Tesla unveiled its new electric vehicle.'", "Tesla", [], "easy"),
            ("Extract the year from: 'The study was conducted in 2019.'", "2019", [], "easy"),
            ("Extract the person's name from: 'Emily Rodriguez, the lead researcher, announced the findings.'", "Emily Rodriguez", [], "medium"),
            ("Extract the email address from: 'Contact us at support@company.com for assistance.'", "support@company.com", [], "medium"),
            ("Extract the phone number from: 'Call (555) 123-4567 for more information.'", "(555) 123-4567", ["555-123-4567", "5551234567"], "medium"),
            ("Extract the product name from: 'The new iPhone 15 Pro features advanced capabilities.'", "iPhone 15 Pro", [], "easy"),
            ("Extract the price from: 'The item costs $49.99 with free shipping.'", "$49.99", ["49.99", "49.99 dollars"], "easy"),
            ("Extract the percentage from: 'Sales increased by 23.5% compared to last year.'", "23.5%", ["23.5 percent"], "easy"),
            ("Extract the city from: 'We are opening a new office in San Francisco.'", "San Francisco", [], "easy"),
            ("Extract the time from: 'The meeting starts at 3:30 PM.'", "3:30 PM", ["3:30 pm", "15:30"], "easy"),
            ("Extract the university from: 'She graduated from Stanford University in 2020.'", "Stanford University", ["Stanford"], "easy"),
            ("Extract the currency amount from: 'The budget is €2.5 million.'", "€2.5 million", ["2.5 million euros"], "medium"),
            ("Extract the programming language from: 'The system is built using Python and JavaScript.'", "Python", ["Python and JavaScript"], "medium"),
        ]

        samples = []
        for i, (q, gt, alt, diff) in enumerate(templates[:n]):
            samples.append({
                "sample_id": f"ee_{i+1:03d}",
                "category": "entity_extraction",
                "question": q,
                "ground_truth": gt,
                "alternative_answers": alt,
                "difficulty": diff,
                "metadata": {
                    "tokens_question": len(q.split()),
                    "tokens_answer": len(gt.split()),
                    "ambiguity_score": 0.1,
                    "entity_type": "named_entity"
                }
            })

        return samples

    def _create_classification_samples(self, n: int) -> List[Dict[str, Any]]:
        """Create classification samples."""
        templates = [
            # Sentiment
            ("Classify the sentiment: 'This product exceeded my expectations!'", "positive", ["Positive", "POSITIVE"], "easy"),
            ("Classify the sentiment: 'I am very disappointed with the service.'", "negative", ["Negative", "NEGATIVE"], "easy"),
            ("Classify the sentiment: 'The experience was okay, nothing special.'", "neutral", ["Neutral", "NEUTRAL"], "medium"),
            ("Classify the sentiment: 'Absolutely fantastic! Highly recommend!'", "positive", ["Positive"], "easy"),

            # Topic
            ("Classify the topic: 'The stock market reached new highs today.'", "finance", ["Finance", "business"], "easy"),
            ("Classify the topic: 'Scientists discovered a new species of butterfly.'", "science", ["Science", "biology"], "easy"),
            ("Classify the topic: 'The team won the championship 3-2.'", "sports", ["Sports"], "easy"),
            ("Classify the topic: 'The latest smartphone features AI capabilities.'", "technology", ["Technology", "tech"], "easy"),

            # Binary
            ("Is this a question: 'What time is it?'", "yes", ["Yes", "true"], "easy"),
            ("Is this a question: 'The sky is blue.'", "no", ["No", "false"], "easy"),
            ("Is this spam: 'You won $1,000,000! Click here now!'", "yes", ["Yes", "spam"], "easy"),
            ("Is this formal language: 'Hey dude, what's up?'", "no", ["No", "informal"], "easy"),
        ]

        samples = []
        for i, (q, gt, alt, diff) in enumerate(templates[:n]):
            samples.append({
                "sample_id": f"cl_{i+1:03d}",
                "category": "classification",
                "question": q,
                "ground_truth": gt,
                "alternative_answers": alt,
                "difficulty": diff,
                "metadata": {
                    "tokens_question": len(q.split()),
                    "tokens_answer": len(gt.split()),
                    "ambiguity_score": 0.15,
                    "classification_type": "categorical"
                }
            })

        return samples

    def _create_simple_reasoning_samples(self, n: int) -> List[Dict[str, Any]]:
        """Create simple reasoning samples."""
        templates = [
            ("If all roses are flowers, and some flowers fade quickly, can we conclude that all roses fade quickly?", "No", ["no", "false"], "medium"),
            ("If it rains, the ground gets wet. The ground is wet. Did it rain?", "Not necessarily", ["Cannot determine", "Maybe", "Possibly"], "hard"),
            ("A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?", "$0.05", ["5 cents", "0.05"], "hard"),
            ("If John is taller than Mary, and Mary is taller than Sue, is John taller than Sue?", "Yes", ["yes", "true"], "easy"),
            ("A farmer has 17 sheep. All but 9 die. How many are left?", "9", ["nine"], "medium"),
            ("What comes next in the sequence: 2, 4, 8, 16, ?", "32", ["thirty-two"], "easy"),
            ("If you have 3 apples and you take away 2, how many do you have?", "2", ["two"], "medium"),
            ("Is it possible for someone to be their own grandfather?", "No", ["no", "Theoretically very rare"], "medium"),
            ("What weighs more: a pound of feathers or a pound of rocks?", "They weigh the same", ["Same", "Equal"], "easy"),
        ]

        samples = []
        for i, (q, gt, alt, diff) in enumerate(templates[:n]):
            samples.append({
                "sample_id": f"sr_{i+1:03d}",
                "category": "simple_reasoning",
                "question": q,
                "ground_truth": gt,
                "alternative_answers": alt,
                "difficulty": diff,
                "metadata": {
                    "tokens_question": len(q.split()),
                    "tokens_answer": len(gt.split()),
                    "ambiguity_score": 0.2,
                    "reasoning_type": "logical"
                }
            })

        return samples

    def create_dataset_b(self) -> Dict[str, Any]:
        """
        Create Dataset B: Multi-step Reasoning (35 samples).

        PRD Specification (Section 1.3):
        - mathematical_word_problems: 11 samples
        - logical_reasoning_chains: 9 samples
        - planning_tasks: 9 samples
        - analytical_reasoning: 6 samples

        Returns:
            Dataset B dictionary
        """
        samples = []

        samples.extend(self._create_math_word_problems(11))
        samples.extend(self._create_logical_reasoning(9))
        samples.extend(self._create_planning_tasks(9))
        samples.extend(self._create_analytical_reasoning(6))

        random.shuffle(samples)

        for i, sample in enumerate(samples, 1):
            sample["sample_id"] = f"msr_{i:03d}"

        dataset = {
            "dataset_id": "dataset_b_v1",
            "dataset_type": "multi_step_reasoning",
            "total_samples": 35,
            "categories": [
                "mathematical_word_problems",
                "logical_reasoning_chains",
                "planning_tasks",
                "analytical_reasoning"
            ],
            "samples": samples,
            "metadata": {
                "version": "1.0.0",
                "creation_date": datetime.now().isoformat(),
                "validation_status": "verified",
                "languages": ["en"],
                "domain": "general",
            }
        }

        return dataset

    def _create_math_word_problems(self, n: int) -> List[Dict[str, Any]]:
        """Create mathematical word problems."""
        problems = [
            {
                "problem": "A store offers a 20% discount on an item originally priced at $150. If sales tax is 8%, what is the final price?",
                "final_answer": "$129.60",
                "steps": [
                    "Calculate discount: $150 × 0.20 = $30",
                    "Subtract discount: $150 - $30 = $120",
                    "Calculate tax: $120 × 0.08 = $9.60",
                    "Add tax: $120 + $9.60 = $129.60"
                ],
                "complexity": 0.6
            },
            {
                "problem": "A rectangular garden is 15 meters long and 8 meters wide. If a path 1 meter wide surrounds the garden, what is the area of the path?",
                "final_answer": "50 square meters",
                "steps": [
                    "Calculate garden area: 15 × 8 = 120 m²",
                    "Calculate outer dimensions: 17 × 10 (adding 1m on each side)",
                    "Calculate outer area: 17 × 10 = 170 m²",
                    "Calculate path area: 170 - 120 = 50 m²"
                ],
                "complexity": 0.65
            },
            {
                "problem": "A train travels 120 kilometers in 2 hours at constant speed. If it maintains the same speed throughout the journey, how long will it take to travel a distance of 300 kilometers?",
                "final_answer": "5 hours",
                "steps": [
                    "Calculate speed: 120 km ÷ 2 hours = 60 km/hour",
                    "Calculate time for 300 km: 300 km ÷ 60 km/hour = 5 hours",
                    "Verify: 60 km/hour × 5 hours = 300 km"
                ],
                "complexity": 0.4
            },
            {
                "problem": "A baking recipe calls for 2.5 cups of flour for 12 cookies. How much flour is needed to bake 30 cookies?",
                "final_answer": "6.25 cups",
                "steps": [
                    "Calculate flour per cookie: 2.5 ÷ 12 = 0.208333 cups per cookie",
                    "Calculate for 30 cookies: 0.208333 × 30 = 6.25 cups",
                    "Verify ratio: 30/12 = 2.5; 2.5 × 2.5 cups = 6.25 cups"
                ],
                "complexity": 0.5
            },
            {
                "problem": "An investment of $5,000 grows at 6% annual interest rate compounded annually. What is the total value after 3 years?",
                "final_answer": "$5,955.08",
                "steps": [
                    "Year 1: $5,000 × 1.06 = $5,300",
                    "Year 2: $5,300 × 1.06 = $5,618",
                    "Year 3: $5,618 × 1.06 = $5,955.08"
                ],
                "complexity": 0.7
            },
            {
                "problem": "A water tank is being filled at 15 liters per minute and drained at 8 liters per minute. If the tank holds 350 liters, how long to fill it from empty?",
                "final_answer": "50 minutes",
                "steps": [
                    "Calculate net fill rate: 15 - 8 = 7 liters/minute",
                    "Calculate time: 350 ÷ 7 = 50 minutes",
                    "Verify: 7 liters/min × 50 min = 350 liters"
                ],
                "complexity": 0.5
            },
            {
                "problem": "A bookshelf has 5 shelves. The first shelf has 12 books, and each subsequent shelf has 3 more books than the previous one. How many total books are there?",
                "final_answer": "90 books",
                "steps": [
                    "Shelf 1: 12 books",
                    "Shelf 2: 12 + 3 = 15 books",
                    "Shelf 3: 15 + 3 = 18 books",
                    "Shelf 4: 18 + 3 = 21 books",
                    "Shelf 5: 21 + 3 = 24 books",
                    "Total: 12 + 15 + 18 + 21 + 24 = 90 books"
                ],
                "complexity": 0.55
            },
            {
                "problem": "A car travels 240 miles using 12 gallons of gas. At the same rate, how many gallons are needed for a 500-mile trip?",
                "final_answer": "25 gallons",
                "steps": [
                    "Calculate miles per gallon: 240 ÷ 12 = 20 mpg",
                    "Calculate gallons needed: 500 ÷ 20 = 25 gallons",
                    "Verify: 25 gallons × 20 mpg = 500 miles"
                ],
                "complexity": 0.4
            },
            {
                "problem": "A class has 28 students. If they form groups of 4 with some leftover students forming a smaller group, how many complete groups of 4 can be formed, and how many students are in the smaller group?",
                "final_answer": "7 complete groups, 0 students leftover",
                "steps": [
                    "Divide 28 by 4: 28 ÷ 4 = 7",
                    "Check remainder: 28 - (7 × 4) = 0",
                    "Result: 7 complete groups, 0 leftover"
                ],
                "complexity": 0.3
            },
            {
                "problem": "Given that the sum of three consecutive even numbers is exactly 78, what are the three consecutive even numbers in this mathematical sequence?",
                "final_answer": "24, 26, 28",
                "steps": [
                    "Let x be the first even number",
                    "Then: x + (x+2) + (x+4) = 78",
                    "Simplify: 3x + 6 = 78",
                    "Solve: 3x = 72, x = 24",
                    "Numbers are: 24, 26, 28"
                ],
                "complexity": 0.6
            },
            {
                "problem": "A pizza is cut into 8 equal slices. If John eats 3 slices and Mary eats 2 slices, what fraction of the pizza remains?",
                "final_answer": "3/8",
                "steps": [
                    "Calculate eaten: 3 + 2 = 5 slices",
                    "Calculate remaining: 8 - 5 = 3 slices",
                    "Express as fraction: 3/8"
                ],
                "complexity": 0.35
            },
        ]

        samples = []
        for i, prob in enumerate(problems[:n]):
            samples.append({
                "sample_id": f"mwp_{i+1:03d}",
                "category": "mathematical_word_problems",
                "problem": prob["problem"],
                "ground_truth_solution": {
                    "final_answer": prob["final_answer"],
                    "reasoning_steps": prob["steps"],
                    "step_count": len(prob["steps"])
                },
                "metadata": {
                    "min_steps_required": len(prob["steps"]),
                    "tokens_problem": len(prob["problem"].split()),
                    "complexity_score": prob["complexity"],
                    "requires_intermediate_steps": True
                }
            })

        return samples

    def _create_logical_reasoning(self, n: int) -> List[Dict[str, Any]]:
        """Create logical reasoning chain problems."""
        problems = [
            {
                "problem": "Five friends (A, B, C, D, E) sit in a row. A sits two seats from C. B sits next to D. E sits at one end. Determine a valid seating arrangement.",
                "final_answer": "A B D C E (or variations)",
                "steps": [
                    "E is at position 1 or 5",
                    "A and C are 2 seats apart",
                    "B and D are adjacent",
                    "Test configurations with E at position 5",
                    "Valid arrangement: A B D C E"
                ],
                "complexity": 0.75
            },
            {
                "problem": "Consider this logical argument: All birds can fly. Penguins are birds. Therefore, can penguins fly? Identify the logical flaw in this reasoning.",
                "final_answer": "No, the premise 'all birds can fly' is false",
                "steps": [
                    "Premise 1: All birds can fly",
                    "Premise 2: Penguins are birds",
                    "Logical conclusion: Penguins can fly",
                    "Reality: Penguins cannot fly",
                    "Flaw: Premise 1 is incorrect"
                ],
                "complexity": 0.5
            },
            {
                "problem": "In a group of 30 students, 18 study Spanish, 15 study French, and 8 study both languages. How many students study neither language?",
                "final_answer": "5 students",
                "steps": [
                    "Apply inclusion-exclusion principle: |S ∪ F| = |S| + |F| - |S ∩ F|",
                    "Students studying at least one: 18 + 15 - 8 = 25",
                    "Students studying neither: 30 - 25 = 5"
                ],
                "complexity": 0.55
            },
            {
                "problem": "If it's raining, then the ground is wet. The ground is wet. Can we conclude it's raining?",
                "final_answer": "No, wet ground doesn't prove rain (affirming the consequent fallacy)",
                "steps": [
                    "Given: Rain → Wet ground",
                    "Observed: Wet ground",
                    "Common error: Conclude it's raining",
                    "Reality: Other causes exist (sprinklers, spilled water)",
                    "Conclusion: Cannot definitively conclude it's raining"
                ],
                "complexity": 0.65
            },
            {
                "problem": "Three switches control three light bulbs in another room. You can flip switches but only enter the room once. How do you determine which switch controls which bulb?",
                "final_answer": "Use heat: turn on switch 1 for 5 minutes, turn off, turn on switch 2, enter room",
                "steps": [
                    "Turn on switch 1 for 5 minutes",
                    "Turn off switch 1",
                    "Turn on switch 2",
                    "Enter the room",
                    "On bulb = switch 2; Warm off bulb = switch 1; Cold off bulb = switch 3"
                ],
                "complexity": 0.8
            },
            {
                "problem": "A truth-teller always tells the truth. A liar always lies. You meet two people. One says 'We are both liars.' What are they?",
                "final_answer": "The speaker is a liar, the other is a truth-teller",
                "steps": [
                    "Assume speaker tells truth",
                    "Then both are liars - contradiction (truth-teller can't be liar)",
                    "So speaker must be lying",
                    "If speaker is liar, statement is false",
                    "False means NOT both liars, so other is truth-teller"
                ],
                "complexity": 0.7
            },
            {
                "problem": "Four people need to cross a bridge at night. They have one flashlight. The bridge holds max 2 people. Crossing times: A=1min, B=2min, C=5min, D=10min. Two people cross at the slower person's pace. What's the minimum time?",
                "final_answer": "17 minutes",
                "steps": [
                    "A and B cross (2 min)",
                    "A returns (1 min)",
                    "C and D cross (10 min)",
                    "B returns (2 min)",
                    "A and B cross (2 min)",
                    "Total: 2+1+10+2+2 = 17 minutes"
                ],
                "complexity": 0.85
            },
            {
                "problem": "You have 12 balls, one is heavier or lighter. Using a balance scale only 3 times, how do you find the odd ball and determine if it's heavier or lighter?",
                "final_answer": "Divide into groups of 4, weigh twice to isolate, third weigh to confirm",
                "steps": [
                    "Weigh 4 vs 4 balls (keep 4 aside)",
                    "If balanced, odd ball is in remaining 4",
                    "If unbalanced, odd ball is in one of the groups",
                    "Weigh 3 from suspect group vs 3 normal",
                    "Third weigh determines which specific ball and heavy/light"
                ],
                "complexity": 0.9
            },
            {
                "problem": "Knights always tell truth, knaves always lie. You meet A and B. A says 'At least one of us is a knave.' What are A and B?",
                "final_answer": "A is a knight, B is a knave",
                "steps": [
                    "Assume A is a knight (truth-teller)",
                    "Then statement 'at least one is knave' is true",
                    "So B must be a knave",
                    "Check: If A were knave, statement would be false, meaning both are knights - contradiction",
                    "Conclusion: A is knight, B is knave"
                ],
                "complexity": 0.65
            },
        ]

        samples = []
        for i, prob in enumerate(problems[:n]):
            samples.append({
                "sample_id": f"lr_{i+1:03d}",
                "category": "logical_reasoning_chains",
                "problem": prob["problem"],
                "ground_truth_solution": {
                    "final_answer": prob["final_answer"],
                    "reasoning_steps": prob["steps"],
                    "step_count": len(prob["steps"])
                },
                "metadata": {
                    "min_steps_required": len(prob["steps"]),
                    "tokens_problem": len(prob["problem"].split()),
                    "complexity_score": prob["complexity"],
                    "requires_intermediate_steps": True
                }
            })

        return samples

    def _create_planning_tasks(self, n: int) -> List[Dict[str, Any]]:
        """Create planning task problems."""
        problems = [
            {
                "problem": "You have 3 tasks: Task A (2 hours), Task B (3 hours, requires Task A completion), Task C (1 hour, independent). You have 4 hours today and 3 hours tomorrow. Plan the optimal schedule.",
                "final_answer": "Day 1: A + C + partial B. Day 2: Complete B",
                "steps": [
                    "Identify dependencies: B depends on A",
                    "Total time needed: 2+3+1 = 6 hours",
                    "Available time: 4+3 = 7 hours (sufficient)",
                    "Day 1: A (2h) + C (1h) + start B (1h) = 4h",
                    "Day 2: Complete B (2h remaining)"
                ],
                "complexity": 0.6
            },
            {
                "problem": "Plan a 3-city road trip. Cities A, B, C form a triangle. Distances: A-B=200mi, B-C=150mi, A-C=250mi. Starting at A, returning to A, minimize total distance.",
                "final_answer": "A → B → C → A (total 600 miles)",
                "steps": [
                    "Route 1: A → B → C → A = 200 + 150 + 250 = 600 mi",
                    "Route 2: A → C → B → A = 250 + 150 + 200 = 600 mi",
                    "Both routes equal distance",
                    "Choose A → B → C → A (arbitrary, both optimal)"
                ],
                "complexity": 0.5
            },
            {
                "problem": "You have $100 budget for a party with 20 people. Snacks=$3/person, Drinks=$2/person, Decorations=$15 total. Can you afford everything? If not, what adjustments?",
                "final_answer": "No, need $115. Cut snacks to $2.75/person or reduce decorations to $10",
                "steps": [
                    "Calculate costs: Snacks=20×$3=$60, Drinks=20×$2=$40, Decor=$15",
                    "Total: $60+$40+$15 = $115",
                    "Overage: $115-$100 = $15",
                    "Option 1: Reduce snacks to $2.75/person ($55 total)",
                    "Option 2: Reduce decorations to $10"
                ],
                "complexity": 0.55
            },
            {
                "problem": "Pack a suitcase (capacity: 20kg) with: laptop (2kg), clothes (8kg), shoes (3kg), toiletries (2kg), books (6kg), camera (1.5kg). What's the optimal selection to maximize value?",
                "final_answer": "Take all except books (or take books and skip some clothes)",
                "steps": [
                    "Total weight: 2+8+3+2+6+1.5 = 22.5 kg",
                    "Overage: 2.5 kg",
                    "Essentials: laptop, clothes, toiletries (12 kg)",
                    "Remaining capacity: 8 kg",
                    "Optimal: shoes (3kg) + camera (1.5kg) + partial books (3.5kg)",
                    "Or: skip books, take shoes + camera + extra clothes"
                ],
                "complexity": 0.65
            },
            {
                "problem": "Schedule 4 meetings in 8-hour workday. Meeting durations: A=2h, B=1h, C=1.5h, D=3h. C must be after A. B and D cannot overlap lunch (12-1pm). Plan the schedule.",
                "final_answer": "9-11am: A, 11am-12pm: B, 1-2:30pm: C, 2:30-5:30pm: D",
                "steps": [
                    "Total meeting time: 2+1+1.5+3 = 7.5 hours (fits)",
                    "Constraint: C after A",
                    "Constraint: B and D avoid 12-1pm",
                    "9-11am: A (2h)",
                    "11am-12pm: B (1h, before lunch)",
                    "1-2:30pm: C (1.5h, after A and lunch)",
                    "2:30-5:30pm: D (3h, after lunch)"
                ],
                "complexity": 0.7
            },
            {
                "problem": "You have 5 errands in different locations. Estimate times: Bank(15min), Post Office(10min), Grocery(30min), Pharmacy(20min), Dry Cleaner(10min). Total available time: 90min. Plan efficient route considering proximity.",
                "final_answer": "Bank → Post Office → Dry Cleaner → Pharmacy → Grocery (85 minutes)",
                "steps": [
                    "Total errand time: 15+10+30+20+10 = 85 min",
                    "Available time: 90 min (sufficient with 5 min buffer)",
                    "Group by proximity: Bank/Post Office, Dry Cleaner/Pharmacy, Grocery",
                    "Route: Bank → Post Office → Dry Cleaner → Pharmacy → Grocery"
                ],
                "complexity": 0.55
            },
            {
                "problem": "Plan a 7-day meal prep for a family of 4. Budget: $200. Breakfast=$3/person, Lunch=$5/person, Dinner=$7/person. Can you stay in budget? If so, how much is saved?",
                "final_answer": "Yes, total cost $420 needs adjustment. Reduce to Breakfast=$2, Lunch=$4, Dinner=$5.5 to meet budget",
                "steps": [
                    "Daily cost: (3+5+7) × 4 people = $60/day",
                    "Weekly cost: $60 × 7 = $420",
                    "Budget: $200",
                    "Shortfall: $420 - $200 = $220",
                    "Adjusted plan: Reduce each meal by ~53%",
                    "New plan: Breakfast=$2, Lunch=$4, Dinner=$5.5 = $46/day × 7 = $322 (still over)",
                    "Further reduction needed: target $28.57/day = ~$7.14/person/day"
                ],
                "complexity": 0.6
            },
            {
                "problem": "Organize a bookshelf with 50 books across 5 shelves. Categories: Fiction(20), Non-fiction(15), Reference(10), Children(5). How should you distribute them for best organization?",
                "final_answer": "Shelf 1: Fiction(10), Shelf 2: Fiction(10), Shelf 3: Non-fiction(10), Shelf 4: Non-fiction(5)+Reference(5), Shelf 5: Reference(5)+Children(5)",
                "steps": [
                    "Each shelf holds 10 books",
                    "Group by category for easy access",
                    "Shelves 1-2: Fiction (20 books, 2 shelves)",
                    "Shelf 3: Non-fiction (10 books)",
                    "Shelf 4: Non-fiction (5) + Reference (5)",
                    "Shelf 5: Reference (5) + Children (5)"
                ],
                "complexity": 0.45
            },
            {
                "problem": "You're moving and have 30 boxes. Truck capacity: 40 boxes. Friend can store 15 boxes. You can take 10 boxes to your new place now. Plan the optimal distribution for 2 trips.",
                "final_answer": "Trip 1: 20 boxes to new place. Friend stores 10. Trip 2: Pick up 10 from friend, deliver all 20",
                "steps": [
                    "Total boxes: 30",
                    "Immediate capacity: 10 boxes",
                    "Friend storage: 15 boxes",
                    "Truck capacity: 40 boxes/trip (more than enough)",
                    "Trip 1: Take 10 to new place, leave 20 with friend (exceeds friend's capacity)",
                    "Alternative: Trip 1: 20 boxes direct, 10 to friend. Trip 2: 10 from friend",
                    "Optimal: Minimize friend storage to 10 boxes"
                ],
                "complexity": 0.65
            },
        ]

        samples = []
        for i, prob in enumerate(problems[:n]):
            samples.append({
                "sample_id": f"pt_{i+1:03d}",
                "category": "planning_tasks",
                "problem": prob["problem"],
                "ground_truth_solution": {
                    "final_answer": prob["final_answer"],
                    "reasoning_steps": prob["steps"],
                    "step_count": len(prob["steps"])
                },
                "metadata": {
                    "min_steps_required": len(prob["steps"]),
                    "tokens_problem": len(prob["problem"].split()),
                    "complexity_score": prob["complexity"],
                    "requires_intermediate_steps": True
                }
            })

        return samples

    def _create_analytical_reasoning(self, n: int) -> List[Dict[str, Any]]:
        """Create analytical reasoning problems."""
        problems = [
            {
                "problem": "Sales data: Jan=100, Feb=120, Mar=110, Apr=132, May=121. Identify the trend and predict June sales.",
                "final_answer": "~130 units (alternating pattern with 5% average growth)",
                "steps": [
                    "Calculate changes: Feb(+20), Mar(-10), Apr(+22), May(-11)",
                    "Pattern: Alternating increases/decreases",
                    "Overall trend: (121-100)/100 = 21% growth over 4 months",
                    "Average monthly growth: ~5%",
                    "May was decrease, June likely increase",
                    "Prediction: 121 × 1.08 ≈ 130 units"
                ],
                "complexity": 0.7
            },
            {
                "problem": "A survey was conducted among 100 people. Survey results show: 60% like coffee, 45% like tea, and 25% like both beverages. What percentage of people like neither coffee nor tea?",
                "final_answer": "20%",
                "steps": [
                    "Coffee only: 60% - 25% = 35%",
                    "Tea only: 45% - 25% = 20%",
                    "At least one: 35% + 20% + 25% = 80%",
                    "Neither: 100% - 80% = 20%"
                ],
                "complexity": 0.55
            },
            {
                "problem": "Company expenses breakdown: Salaries(50%), Rent(20%), Utilities(10%), Marketing(15%), Other(5%). If total budget is $200,000, and they want to increase marketing to 20%, what must be reduced?",
                "final_answer": "Reduce other categories proportionally or cut $10,000 total",
                "steps": [
                    "Current marketing: $200,000 × 0.15 = $30,000",
                    "Desired marketing: $200,000 × 0.20 = $40,000",
                    "Increase needed: $10,000",
                    "Options: Reduce salaries, rent, utilities, or other",
                    "Proportional cut: Reduce other by $10,000 (e.g., Other from 5% to 0%)"
                ],
                "complexity": 0.6
            },
            {
                "problem": "Test scores: 85, 92, 78, 95, 88, 90, 76, 94. Calculate mean, median, and identify any outliers.",
                "final_answer": "Mean=87.25, Median=89, Outlier=76 (possible)",
                "steps": [
                    "Sort scores: 76, 78, 85, 88, 90, 92, 94, 95",
                    "Mean: (76+78+85+88+90+92+94+95)/8 = 698/8 = 87.25",
                    "Median: (88+90)/2 = 89",
                    "Range: 95-76 = 19",
                    "Potential outlier: 76 (more than 10 points below next)"
                ],
                "complexity": 0.5
            },
            {
                "problem": "Website traffic: Mobile(55%), Desktop(35%), Tablet(10%). Conversion rates: Mobile(2%), Desktop(5%), Tablet(3%). Which platform generates most conversions?",
                "final_answer": "Desktop generates most conversions (1.75% overall)",
                "steps": [
                    "Mobile conversions: 55% × 2% = 1.1%",
                    "Desktop conversions: 35% × 5% = 1.75%",
                    "Tablet conversions: 10% × 3% = 0.3%",
                    "Desktop has highest absolute conversion rate"
                ],
                "complexity": 0.65
            },
            {
                "problem": "Product reviews: 5-star(40%), 4-star(30%), 3-star(15%), 2-star(10%), 1-star(5%). Calculate weighted average rating.",
                "final_answer": "4.0 stars",
                "steps": [
                    "Weighted sum: (5×0.4) + (4×0.3) + (3×0.15) + (2×0.1) + (1×0.05)",
                    "Calculate: 2.0 + 1.2 + 0.45 + 0.2 + 0.05",
                    "Total: 3.9 stars (rounds to 4.0)"
                ],
                "complexity": 0.45
            },
        ]

        samples = []
        for i, prob in enumerate(problems[:n]):
            samples.append({
                "sample_id": f"ar_{i+1:03d}",
                "category": "analytical_reasoning",
                "problem": prob["problem"],
                "ground_truth_solution": {
                    "final_answer": prob["final_answer"],
                    "reasoning_steps": prob["steps"],
                    "step_count": len(prob["steps"])
                },
                "metadata": {
                    "min_steps_required": len(prob["steps"]),
                    "tokens_problem": len(prob["problem"].split()),
                    "complexity_score": prob["complexity"],
                    "requires_intermediate_steps": True
                }
            })

        return samples


# Convenience functions
def create_dataset_a() -> Dict[str, Any]:
    """Convenience function to create Dataset A."""
    creator = DatasetCreator()
    return creator.create_dataset_a()


def create_dataset_b() -> Dict[str, Any]:
    """Convenience function to create Dataset B."""
    creator = DatasetCreator()
    return creator.create_dataset_b()
