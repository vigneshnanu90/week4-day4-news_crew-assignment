# Crew AI News Agent

A two-agent [CrewAI](https://www.crewai.com/) crew that finds today's top 3 tech news stories, writes a 5-line newsletter, and publishes everything as a newspaper-style PDF with headlines, summaries and images.

## How it works

| Step | Agent | Tools | Output |
|------|-------|-------|--------|
| 1. Research | **Tech News Researcher** | `SerperDevTool` (news search) | Top 3 stories: headline, summary, source URL, image URL |
| 2. Write | **Newsletter Writer** | none | A newsletter of **exactly 5 lines** |
| 3. Publish | reportlab | n/a | A one-page newspaper PDF |

- The tasks run in order with `Process.sequential`.
- The Writer cannot search. It only receives the Researcher's result through `context=[research_task]`.
- The 5 lines are: 1 headline, 3 story lines (one per story), 1 sign-off. A task guardrail checks the line count and makes the Writer retry if it is wrong.
- If the Researcher returns fewer than 3 stories, the crew is re-run (up to 3 attempts).
- Story images come from the search result's `imageUrl`. If that fails, the script falls back to the article's `og:image`. If neither works, the story is printed without an image.

## Requirements

- Python 3.10 or newer
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Serper API key](https://serper.dev) (note: this is serper.dev, not SerpAPI)

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. Install dependencies
python -m pip install crewai crewai-tools reportlab pillow requests python-dotenv
```

Create a `.env` file in the project folder (see `.env.example`):

```
OPENAI_API_KEY=sk-...
SERPER_API_KEY=your_serper_key
```

Use no quotes and no spaces around `=`. Never commit this file.

## Usage

```bash
# Full run: research + write + PDF
python news_crew.py

# Choose the output file name
python news_crew.py -o my_newspaper.pdf

# Preview the PDF layout with sample data (no API keys needed)
python news_crew.py --demo
```

The result is saved as `tech_times_YYYY-MM-DD.pdf` in the project folder, and the 5-line newsletter is also printed in the terminal.

## Project structure

```
crew-ai-news-agent/
├── news_crew.py     # agents, tasks, crew and PDF generation (single file)
├── .env             # your API keys (not committed)
├── .env.example     # template for .env
├── .gitignore
└── README.md
```

## Assignment checklist

- [x] All code in a single file, `news_crew.py`
- [x] Two agents, each with its own role, goal and backstory
- [x] Researcher carries `SerperDevTool`; Writer carries no tools
- [x] Two tasks: find the top 3 stories, then write the newsletter
- [x] `process=Process.sequential`
- [x] Researcher's result passed to the Writer's task via `context`
- [x] Writer's `expected_output` states the output is exactly 5 lines
- [x] `verbose=True`
- [x] Both keys read from environment variables

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `403 Unauthorized` from `google.serper.dev` | The Serper key is wrong, expired, out of credits, or from the wrong service. Generate a key at serper.dev and check the `.env` formatting. |
| `ModuleNotFoundError: No module named 'reportlab'` | Run `python -m pip install reportlab pillow requests python-dotenv` inside the activated venv. |
| `Missing environment variable(s)` | Make sure `.env` is in the folder you run the script from, with both keys set. |
| A story has no image | Some sites block image downloads. This is expected, and the layout still works. |
| Fewer than 3 stories in the PDF | The script retries automatically. If it still happens, run it again or check your Serper credits. |

## Notes

- Built-in PDF fonts cannot draw emoji or non-Latin characters, so those are dropped from the text.
- The generated PDF is compiled by an AI. Check the source links before sharing.