# Glass Firing Viewer

Expected repository layout:

```text
glass_cut_predictor/
├── data/
│   └── Brandt001_14_09_2026/
│       ├── metadata.json
│       └── processed/
│           ├── before_calibrated_split/
│           └── after_calibrated_split_aligned/
└── frontend/
```

The frontend reads `../data/` directly. It does not copy or modify the dataset.

## Start locally

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Vercel

Import the GitHub repository and set the Vercel Root Directory to `frontend`.

`next.config.ts` includes the repository-level `data/` directory in Next.js output-file tracing. For a small dataset this is convenient. If the dataset later becomes large, move it to object storage.
