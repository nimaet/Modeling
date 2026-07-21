# ROM Derivation Note

`main.tex` organizes the modeling and analysis chain for collaborators:

1. physical piezoelectric beam with two electrical paths,
2. assumptions leading to the two-DOF reduced-order model,
3. normalization,
4. complex-amplitude/slow-flow derivation,
5. steady-state cubic, hysteresis, and stability check,
6. links to the cleaned MATLAB simulation archive.

Build with:

```bash
latexmk -pdf main.tex
```

The note intentionally keeps provenance visible and does not overwrite the old
Word documents in `Old_sources/`.
