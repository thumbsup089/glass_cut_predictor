"use client";

import {
  useEffect,
  useMemo,
  useState
} from "react";

import StlViewer from "@/components/StlViewer";


/* =========================================================
   TYPES
========================================================= */

type Brand = {
  folder: string;
  label: string;
};


type Piece = {
  id: string;
  stl?: string;
  glass_thickness_mm?: number;

  before: {
    crop: string;
  };

  after: {
    crop: string;
  };
};


type Calibration = {
  pixels_per_mm?: number;
  mean_error_mm?: number;
  max_error_mm?: number;
};


type SourceImage = {
  id?: string;
  phase?: "before" | "after";
  calibration?: Calibration;
};


type Metadata = {
  brand_id?: string;
  date?: string;

  program_csv?: string;

  notes?: string[];

  images?: SourceImage[];

  fired_pieces?: Piece[];
};


type ProgramRow = {
  seg: string;
  rampe: string;
  ziel: string;
  haltezeit: string;
};


type Mode =
  | "side"
  | "overlay";


/* =========================================================
   PATH HELPERS
========================================================= */

const apiFile = (
  brand: string,
  rel: string
) =>
  `/api/files/${encodeURIComponent(brand)}/${rel
    .replace(/\\/g, "/")
    .split("/")
    .filter(Boolean)
    .map(encodeURIComponent)
    .join("/")}`;


const basename = (
  p: string
) =>
  p
    .replace(/\\/g, "/")
    .split("/")
    .pop() ?? p;


/* =========================================================
   MAIN COMPONENT
========================================================= */

export default function ExperimentBrowser() {

  const [brands, setBrands] =
    useState<Brand[]>([]);

  const [brand, setBrand] =
    useState("");

  const [metadata, setMetadata] =
    useState<Metadata | null>(null);

  /*
   * Empty pieceId means:
   *
   *     OVERVIEW
   */
  const [pieceId, setPieceId] =
    useState("");

  const [mode, setMode] =
    useState<Mode>("side");

  const [opacity, setOpacity] =
    useState(0.5);

  const [error, setError] =
    useState("");


  /* ---------------------------------------------------------
     LOAD BRANDS
  --------------------------------------------------------- */

  useEffect(() => {

    fetch(
      "/api/brands",
      {
        cache: "no-store"
      }
    )

      .then((r) => {

        if (!r.ok) {
          throw new Error(
            "Could not load Brands."
          );
        }

        return r.json();
      })

      .then((d) => {

        const b =
          d.brands ?? [];

        setBrands(b);

        if (b.length) {
          setBrand(
            b[0].folder
          );
        }
      })

      .catch((e) =>
        setError(e.message)
      );

  }, []);


  /* ---------------------------------------------------------
     LOAD METADATA
  --------------------------------------------------------- */

  useEffect(() => {

    if (!brand) {
      return;
    }

    setMetadata(null);

    /*
     * Every time the Brand changes:
     * open Overview first.
     */
    setPieceId("");

    setError("");

    fetch(
      `/api/brands/${encodeURIComponent(brand)}`,
      {
        cache: "no-store"
      }
    )

      .then((r) => {

        if (!r.ok) {
          throw new Error(
            "Could not load metadata."
          );
        }

        return r.json();
      })

      .then(
        (d: Metadata) => {

          setMetadata(d);

          /*
           * Do NOT automatically select
           * first fired piece.
           *
           * pieceId stays empty,
           * therefore Overview is shown.
           */
        }
      )

      .catch((e) =>
        setError(e.message)
      );

  }, [brand]);


  /* ---------------------------------------------------------
     PIECES
  --------------------------------------------------------- */

  const pieces =
    metadata?.fired_pieces ?? [];


  const piece = useMemo(

    () =>
      pieces.find(
        (p) =>
          p.id === pieceId
      ) ?? null,

    [
      pieces,
      pieceId
    ]

  );


  /* ---------------------------------------------------------
     IMAGE URLS
  --------------------------------------------------------- */

  const beforeUrl =
    piece
      ? apiFile(
        brand,
        piece.before.crop
      )
      : "";


  const afterUrl =
    piece
      ? apiFile(
        brand,
        `processed/after_calibrated_split_aligned/${basename(
          piece.after.crop
        )}`
      )
      : "";

  const stlUrl =
    piece && piece.stl
      ? apiFile(
        brand,
        piece.stl
      )
      : "";


  /* =========================================================
     UI
  ========================================================= */

  return (

    <main className="shell">

      {/* =====================================================
          SIDEBAR
      ===================================================== */}

      <aside className="sidebar">

        <div className="title">
          Glass Firing
        </div>


        {/* BRAND SELECT */}

        <label>
          Brand
        </label>

        <select
          value={brand}
          onChange={(e) =>
            setBrand(
              e.target.value
            )
          }
        >

          {!brands.length && (
            <option>
              No Brands found
            </option>
          )}

          {brands.map(
            (b) => (

              <option
                key={b.folder}
                value={b.folder}
              >
                {b.label}
              </option>

            )
          )}

        </select>


        {/* OVERVIEW */}

        <div className="pieceHeading">
          Brand
        </div>

        <button
          className={
            pieceId === ""
              ? "piece active"
              : "piece"
          }
          onClick={() =>
            setPieceId("")
          }
        >
          Overview
        </button>


        {/* FIRED PIECES */}

        <div className="pieceHeading">
          Fired Pieces
        </div>

        <div className="pieceList">

          {pieces.map(
            (p) => (

              <button
                key={p.id}

                className={
                  p.id === pieceId
                    ? "piece active"
                    : "piece"
                }

                onClick={() =>
                  setPieceId(
                    p.id
                  )
                }
              >
                {p.id}
              </button>

            )
          )}

        </div>


        {error && (
          <div className="error">
            {error}
          </div>
        )}

      </aside>


      {/* =====================================================
          MAIN VIEW
      ===================================================== */}

      <section className="viewer">

        {/* LOADING */}

        {!metadata && brand && !error && (

          <div className="empty">
            Loading Brand…
          </div>

        )}


        {/* OVERVIEW */}

        {metadata &&
          pieceId === "" && (

            <BrandOverview
              brand={brand}
              metadata={metadata}
            />

          )}


        {/* FIRED PIECE */}

        {metadata &&
          pieceId !== "" &&
          piece && (

            <>

              <header>

                <div>

                  <div className="small">
                    FIRED PIECE
                  </div>

                  <h1>
                    {piece.id}
                  </h1>

                  <div className="meta">

                    Glass thickness:{" "}

                    {piece.glass_thickness_mm
                      ?? "—"}{" "}

                    mm

                  </div>

                </div>


                {/* VIEW MODE */}

                <div className="tabs">

                  <button
                    className={
                      mode === "side"
                        ? "selected"
                        : ""
                    }

                    onClick={() =>
                      setMode("side")
                    }
                  >
                    Side by side
                  </button>


                  <button
                    className={
                      mode === "overlay"
                        ? "selected"
                        : ""
                    }

                    onClick={() =>
                      setMode(
                        "overlay"
                      )
                    }
                  >
                    Overlay
                  </button>

                </div>

              </header>


              {/* SIDE BY SIDE */}

              {mode === "side" ? (

                <div className="grid">

                  <ImageBox
                    label="BEFORE"
                    src={beforeUrl}
                  />

                  <ImageBox
                    label="AFTER"
                    src={afterUrl}
                  />

                </div>

              ) : (

                /* OVERLAY */

                <div>

                  <div className="slider">

                    <span>
                      After opacity
                    </span>

                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={opacity}

                      onChange={(e) =>
                        setOpacity(
                          Number(
                            e.target.value
                          )
                        )
                      }
                    />

                    <span>
                      {Math.round(
                        opacity * 100
                      )}
                      %
                    </span>

                  </div>


                  <div className="overlay">

                    <img
                      src={beforeUrl}
                      alt="Before"
                    />

                    <img
                      src={afterUrl}
                      alt="After aligned"
                      className="after"
                      style={{
                        opacity
                      }}
                    />

                  </div>

                </div>

              )}

              {stlUrl && (
                <section style={{ marginTop: "32px" }}>
                  <div className="small">
                    NEGATIVE — 3D MODEL
                  </div>

                  <div style={{ marginTop: "8px" }}>
                    <StlViewer
                      src={stlUrl}
                      height={360}
                    />
                  </div>
                </section>
              )}

            </>

          )}

      </section>

    </main>

  );
}


/* =========================================================
   BRAND OVERVIEW
========================================================= */

function BrandOverview({
  brand,
  metadata
}: {
  brand: string;
  metadata: Metadata;
}) {

  const pieces =
    metadata.fired_pieces ?? [];

  const images =
    metadata.images ?? [];

  const notes =
    metadata.notes ?? [];


  /* ---------------------------------------------------------
     THICKNESS COUNTS
  --------------------------------------------------------- */

  const thicknessCounts =
    pieces.reduce<
      Record<string, number>
    >(
      (
        result,
        piece
      ) => {

        const thickness =
          piece.glass_thickness_mm;

        if (
          thickness != null
        ) {

          const key =
            `${thickness} mm`;

          result[key] =
            (
              result[key] ?? 0
            ) + 1;
        }

        return result;

      },
      {}
    );


  /* ---------------------------------------------------------
     IMAGE COUNTS
  --------------------------------------------------------- */

  const beforeImages =
    images.filter(
      (image) =>
        image.phase === "before"
    ).length;


  const afterImages =
    images.filter(
      (image) =>
        image.phase === "after"
    ).length;


  /* ---------------------------------------------------------
     CALIBRATION SUMMARY
  --------------------------------------------------------- */

  const calibrationErrors =
    images
      .map(
        (image) =>
          image.calibration
            ?.mean_error_mm
      )
      .filter(
        (
          value
        ): value is number =>
          typeof value === "number"
      );


  const averageCalibrationError =
    calibrationErrors.length
      ? calibrationErrors.reduce(
        (sum, value) =>
          sum + value,
        0
      ) /
      calibrationErrors.length
      : null;


  /* =========================================================
     OVERVIEW UI
  ========================================================= */

  return (

    <div className="brandOverview">

      <div className="small">
        BRAND OVERVIEW
      </div>


      <h1>
        {metadata.brand_id
          ?? brand}
      </h1>


      {/* SUMMARY CARDS */}

      <div className="overviewGrid">

        <OverviewCard
          label="DATE"
          value={
            formatDate(
              metadata.date
            )
          }
        />


        <OverviewCard
          label="FIRED PIECES"
          value={
            String(
              pieces.length
            )
          }
        />


        {Object.entries(
          thicknessCounts
        ).map(
          (
            [
              thickness,
              count
            ]
          ) => (

            <OverviewCard
              key={thickness}

              label={
                `${thickness} GLASS`
              }

              value={
                `${count} ${count === 1
                  ? "piece"
                  : "pieces"
                }`
              }
            />

          )
        )}


        {images.length > 0 && (

          <OverviewCard
            label="SOURCE IMAGES"
            value={
              `${beforeImages} before / ${afterImages} after`
            }
          />

        )}


        {averageCalibrationError
          != null && (

            <OverviewCard
              label="AVG. CALIBRATION ERROR"
              value={
                `${averageCalibrationError.toFixed(
                  3
                )} mm`
              }
            />

          )}

      </div>


      {/* FIRING PROGRAM */}

      <FiringProgram
        brand={brand}
        programCsv={
          metadata.program_csv
        }
      />


      {/* NOTES */}

      {notes.length > 0 && (

        <section className="overviewSection">

          <div className="small">
            NOTES
          </div>

          <div className="notesBox">

            {notes.map(
              (
                note,
                index
              ) => (

                <div
                  key={index}
                  className="note"
                >
                  {note}
                </div>

              )
            )}

          </div>

        </section>

      )}

    </div>

  );
}


/* =========================================================
   OVERVIEW CARD
========================================================= */

function OverviewCard({
  label,
  value
}: {
  label: string;
  value: string;
}) {

  return (

    <div className="overviewCard">

      <div className="small">
        {label}
      </div>

      <strong>
        {value}
      </strong>

    </div>

  );
}


/* =========================================================
   FIRING PROGRAM
========================================================= */

function FiringProgram({
  brand,
  programCsv
}: {
  brand: string;
  programCsv?: string;
}) {

  const [
    rows,
    setRows
  ] =
    useState<ProgramRow[]>([]);


  const [
    loading,
    setLoading
  ] =
    useState(false);


  const [
    failed,
    setFailed
  ] =
    useState(false);


  useEffect(() => {

    if (!programCsv) {

      setRows([]);
      setFailed(false);

      return;
    }


    setLoading(true);
    setFailed(false);


    const url =
      apiFile(
        brand,
        programCsv
      );


    fetch(url)

      .then((response) => {

        if (!response.ok) {

          throw new Error(
            "Program CSV could not be loaded."
          );

        }

        return response.text();

      })

      .then((text) => {

        setRows(
          parseProgramCsv(
            text
          )
        );

      })

      .catch(() => {

        setFailed(true);
        setRows([]);

      })

      .finally(() => {

        setLoading(false);

      });

  }, [
    brand,
    programCsv
  ]);


  return (

    <section className="overviewSection">

      <div className="small">
        FIRING PROGRAM
      </div>


      {!programCsv && (

        <div className="overviewMessage">
          No firing program linked.
        </div>

      )}


      {programCsv &&
        loading && (

          <div className="overviewMessage">
            Loading firing program…
          </div>

        )}


      {programCsv &&
        failed && (

          <div className="overviewMessage">
            Could not load{" "}
            {programCsv}.
          </div>

        )}


      {programCsv &&
        !loading &&
        !failed &&
        rows.length > 0 && (

          <div className="programTableWrapper">

            <table className="programTable">

              <thead>

                <tr>
                  <th>Seg</th>
                  <th>Rampe</th>
                  <th>Ziel</th>
                  <th>Haltezeit</th>
                </tr>

              </thead>


              <tbody>

                {rows.map(
                  (
                    row,
                    index
                  ) => (

                    <tr key={index}>

                      <td>
                        {row.seg}
                      </td>

                      <td>
                        {row.rampe}
                      </td>

                      <td>
                        {row.ziel}
                      </td>

                      <td>
                        {row.haltezeit}
                      </td>

                    </tr>

                  )
                )}

              </tbody>

            </table>

          </div>

        )}

    </section>

  );
}


/* =========================================================
   CSV PARSER
========================================================= */

function parseProgramCsv(
  text: string
): ProgramRow[] {

  const cleaned =
    text
      .replace(/^\uFEFF/, "")
      .trim();


  if (!cleaned) {
    return [];
  }


  const lines =
    cleaned
      .split(/\r?\n/)
      .filter(Boolean);


  if (
    lines.length < 2
  ) {
    return [];
  }


  /*
   * Support comma or semicolon CSV.
   */

  const delimiter =
    lines[0].includes(";")
      ? ";"
      : ",";


  return lines
    .slice(1)
    .map((line) => {

      const columns =
        line
          .split(delimiter)
          .map(
            (value) =>
              value
                .trim()
                .replace(
                  /^"(.*)"$/,
                  "$1"
                )
          );


      return {
        seg:
          columns[0] ?? "",

        rampe:
          columns[1] ?? "",

        ziel:
          columns[2] ?? "",

        haltezeit:
          columns[3] ?? ""
      };

    });
}


/* =========================================================
   IMAGE BOX
========================================================= */

function ImageBox({
  label,
  src
}: {
  label: string;
  src: string;
}) {

  const [
    failed,
    setFailed
  ] =
    useState(false);


  useEffect(() => {

    setFailed(false);

  }, [src]);


  return (

    <div>

      <div className="small">
        {label}
      </div>


      <div className="imageBox">

        {failed ? (

          <div className="empty">
            Image could not be loaded.
          </div>

        ) : (

          <img
            src={src}
            alt={label}

            onError={() =>
              setFailed(true)
            }
          />

        )}

      </div>

    </div>

  );
}


/* =========================================================
   DATE
========================================================= */

function formatDate(
  value?: string
) {

  if (!value) {
    return "—";
  }


  const match =
    value.match(
      /^(\d{4})-(\d{2})-(\d{2})$/
    );


  if (!match) {
    return value;
  }


  return (
    `${match[3]}.` +
    `${match[2]}.` +
    `${match[1]}`
  );
}