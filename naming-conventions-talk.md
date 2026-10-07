# Naming Conventions & Philosophies: Talk Notes

**Audience:** young data professionals
**Slot:** 45-minute conference talk
**Status:** exploration only. Nothing is built yet (no slides, no demo).

> **Verify before presenting.** Everything below is from memory, not checked against sources. Treat all specifics as unverified: identifier limits, engine case-folding, Power BI and Fabric behavior, and PEP 8 wording. Check each against official docs before it goes on a slide. Don't put any statistic about AI performance on a slide unless you can cite a source.

---

## 1. Thesis

Names are the interface between your work and every future reader. That reader is now a human reviewer, a future maintainer, **and a model**.

> "Naming used to be a courtesy to your teammates. Now it's also an interface to your tools."

Supporting claims:

- Code is read far more than it is written (PEP 8 says so directly; *Clean Code* puts the ratio at well over 10:1, which is a rhetorical estimate, not a measurement).
- A style guide moves effort from the many readers to the one writer.
- In data work the imbalance is worse. Queries and models are read during review, debugging, incidents, audits, onboarding, and every time someone asks "where does this number come from?"
- AI makes code cheaper to write, so **reading and reviewing become the bottleneck**.

---

## 2. Proposed 45-minute timeline

| Min | Section |
|---|---|
| 0-3 | **Hook:** `final_v2_FINAL_clean_new.csv`, `df2`, `tmp3`, `col_17`. Ask who has written one. |
| 3-7 | **Why naming matters more in data work** |
| 7-14 | **Self-documenting code** |
| 14-20 | **Conventions in the data stack:** casing (2 min) and word order (4 min) |
| 20-30 | **Philosophies:** types in names, Hungarian history, PEP 8 |
| 30-37 | **SQL and database standards, limits, Power BI / Fabric** |
| 37-42 | **Live exercise:** rename a bad schema and rewrite a bad query in groups |
| 42-45 | **Takeaways, style guides in the age of AI, Q&A** |

More material is collected below than fits in 45 minutes. See [Open questions](#13-open-questions-and-things-to-cut).

---

## 3. Why naming matters more in data work

- Notebooks get handed off.
- The schema outlives its author.
- Column names become **API contracts** for dashboards, models, and other teams.
- Renaming a warehouse column is a breaking change for everything downstream.

---

## 4. Self-documenting code

**The idea:** names and structure explain *what* the code does, so comments are free to explain *why*.

### Techniques (with data examples)

- **Names over comments.** Replace `-- remove test accounts` above `WHERE acct_type <> 3` with a clearly named column or CTE.
- **Named constants over magic numbers.** `status = 3` becomes a lookup-table join or an `accepted_values` test.
- **CTEs as named steps.** This is SQL's version of small functions. Compare one 200-line nested subquery with CTEs named `orders_last_90_days`, `first_purchase_per_customer`, and `repeat_customers`.
- **Small, single-purpose units.** One transform per model or function, named for what it returns.
- **Types and constraints as documentation.** `NOT NULL`, `CHECK`, foreign keys, and dbt tests state the rules in an enforced form that can't drift.

### The honest limits

- Names can't carry **why**: the business rule, the workaround for a bad upstream source, the performance trade-off. Those still need comments.
- Self-documenting is **not** no-documentation. Data still needs a dictionary (`COMMENT ON COLUMN`, dbt `description:`), because a column name can't hold the definition of "active user".
- Over-long names are a failure mode. `customer_total_lifetime_net_revenue_after_refunds_usd` signals that the concept needs a definition and a shorter name.
- Tests are documentation that stays honest. A comment can go stale, but a failing test can't.

**Slide idea:** the same query before and after. Before: nested subqueries with aliases `a`, `b`, `t1` and comments explaining each. After: named CTEs with one comment explaining a single *why*.

**Tie-back:** self-documenting code is "make wrong code look wrong" again. Apps Hungarian does it through names, and constraints and tests do it through enforcement.

---

## 5. Mechanical conventions in the data stack

### Casing by ecosystem

| Context | Typical style |
|---|---|
| SQL, pandas, Python | `snake_case` |
| JSON, JS | `camelCase` |
| Constants | `UPPER_CASE` |
| Classes | `CapWords` |

- Quoted identifiers (`"UserName"`) become case-sensitive forever in Snowflake and Postgres.
- Portable default: **lowercase `snake_case`, no spaces, never quote.**

### Table and column conventions

- Singular or plural? `user` vs `users`. Pick one and keep it.
- Layer prefixes: `stg_`, `int_`, `dim_`, `fct_` (dbt and Kimball style). These encode a role or layer, not a datatype.
- IDs: `id` vs `user_id`. dbt style: `<object>_id`.
- Timestamps `<event>_at`, dates `<event>_date`, booleans `is_` / `has_`.

---

## 6. Word order: `cleaned_string` or `string_clean`?

| Style | Pattern | Example | Reads like |
|---|---|---|---|
| Natural language | modifier + noun | `cleaned_string`, `total_revenue` | English |
| Noun-first (hierarchical) | noun + modifier | `string_clean`, `revenue_total` | A path or index |

### Case for noun-first

- **Alphabetical grouping.** `revenue_gross`, `revenue_net`, `revenue_refunded` cluster together in autocomplete, column lists, and file trees. The natural-order versions scatter.
- **It matches ISO/IEC 11179** (object class, then property, then representation term): `customer_birth_date`.
- **Prefix search is how people look things up.** Typing `order_` and seeing everything about orders helps in a 300-column table.
- **Precedent:** namespaces and hierarchies (`user.name`, `user.email`).

### Case for natural order

- **It reads aloud.** `cleaned_string` sounds like something you'd say, and `string_clean` sounds like a command.
- **Fewer ambiguity traps.** `string_clean` could be a clean string, a function that cleans, or a flag.
- **Booleans read well:** `is_active`, `has_orders`.

### Rule of thumb

> **Order by what the reader is scanning for.** Lead with what they'll search or group by (usually the entity), and end with what varies (state, unit, or aggregate).

- **Locally** (function or notebook, read top to bottom): natural English (`cleaned_names`, `filtered_orders`).
- **Shared schema, warehouse, or API** (names get sorted and searched): entity first, qualifier last (`order_total_usd`, `order_created_at`, `order_status`).
- **Be consistent.** Mixing `total_revenue` and `revenue_net` in one table is worse than either convention.

### Data-specific pitfalls

- **Verb vs noun.** `clean_string` could be the function or the result. Functions are verbs (`clean_string()`), values are nouns or past participles (`cleaned_string`).
- **Pipeline stage as prefix** (`raw_`, `stg_`, `cleaned_`) so lists sort by layer, which is the dbt approach.
- **Unit and time suffixes go last:** `revenue_usd`, `created_at_utc`. Almost every guide agrees on this one.
- **Aggregations:** `avg_order_value` reads naturally, but `order_value_avg` groups with `order_value_min` and `order_value_max`.
- **Avoid negated booleans.** Use `is_active` and negate in logic, not `is_not_active`.

**Slide idea:** a live-sort demo. Take the same ten columns named both ways and alphabetize them side by side. Noun-first clusters, and natural order scatters. Then read each name out loud, where natural order wins. The tension between *scannable* and *speakable* is the point.

---

## 7. Philosophies: what a name should carry

- **Intention-revealing names** (*Clean Code*): the name says why it exists, what it does, and how it's used.
- **Ubiquitous language** (Domain-Driven Design): names come from the business, and code and conversation share one vocabulary. `customer` vs `client` vs `account` means three things to three teams.
- **Short and local vs long and global** (Go): scope sets name length. `df` and `i` are fine in a 5-line scope, but a warehouse column must be self-explanatory.
- **Explicit over terse** (Swift API guidelines): names should read like English at the call site.
- **Convention over configuration** (Rails): the name carries behavior (`User` maps to the `users` table).
- **Consistency over cleverness:** one concept, one name, everywhere (`customer_id`, not `cust_id` here and `client_key` there).

---

## 8. Datatypes in variable names (`username_string`)

### The case against

- **Redundant.** The schema or type system already says it, in one authoritative place.
- **It lies when the type changes.** `age_int` becomes a decimal, and now every query and dashboard carries a wrong name, or you do a breaking rename.
- **Noise.** `username_string` reads worse than `username`, and `customer_list` may really be a set, Series, or DataFrame.
- **Leaks implementation.** Callers shouldn't care whether it's a string or a varchar.

### The case for (it's not always wrong)

- **Untyped or weakly typed contexts:** CSVs, JSON blobs, wide pandas frames where everything is `object`, spreadsheet exports, NoSQL documents. `signup_date_str` flags that the value hasn't been parsed yet.
- **Same value, multiple representations:** `order_date_str` vs `order_date`, `price_cents` vs `price_usd`, `created_at_utc` vs `created_at_local`. Here the suffix is really a **unit, format, or state**, which is valuable.
- **Ambiguous containers:** `users_df`, `users_list`, and `users_dict` in one function.
- **Boundary code:** raw and staging layers where you deliberately name pre-cast columns (`raw_amount`, `amount_text`).

### Verdict

> **Encode meaning, not machine types.** If a suffix tells the reader something the type system can't (unit, timezone, format, pipeline stage), keep it. If it only repeats what a type annotation or schema already says, drop it.

**Slide idea:** a 2x2 of "type info is redundant / valuable" against "context is typed / untyped", with examples in each quadrant.

### Sharper version of the rule

A type prefix is justified when the type is **not visible or not trustworthy where the name is read**. See the Fabric item-prefix case in [section 11](#11-power-bi-and-fabric).

---

## 9. Hungarian notation: Apps vs Systems

**Charles Simonyi** (Xerox PARC, later Microsoft Word and Excel) proposed prefixing names with a short tag. It's called Hungarian partly because he's Hungarian and partly because the names look foreign.

### Apps Hungarian: the prefix encodes *meaning*

The prefix describes a *kind* in the program's domain, not a machine type. Two values can both be plain integers and still be kinds that must never mix.

- `rw` = row, `col` = column. Both `int`, but `rw = col` is visibly wrong.
- `us` = unsafe string (raw user input), `s` = safe string (already escaped). `sName = usName` without an encoding step looks wrong.
- `dx` / `dy` = width or height *difference*, as opposed to `x` / `y` positions.

Joel Spolsky's essay **"Making Wrong Code Look Wrong"** is the best-known explanation.

### Systems Hungarian: the version people remember

When the convention spread through the Windows API docs, it was misread. Simonyi's paper used the word "type" to mean a *kind* of thing, and Windows programmers read it as machine type:

- `lpszName` = long pointer to a zero-terminated string
- `dwFlags` = double word
- `bIsActive` = boolean
- `iCount` = int

The prefix now repeated what the compiler already knew. It stopped catching logic errors and became noise, and when a type changed, every name was wrong or needed a risky rename.

| | Apps Hungarian | Systems Hungarian |
|---|---|---|
| Prefix says | domain meaning (row vs column, safe vs unsafe) | storage type (int, string, pointer) |
| Compiler already knows it? | no | yes |
| Verdict | valuable | redundant |

**Data-world equivalent of the apps style:** suffixes like `_usd` vs `_cents`, `_utc` vs `_local`, or `raw_` vs cleaned. They carry meaning the type system can't express.

**Talk structure idea:** open the types-in-names section with this story: one good idea, one misreading, and a lesson about what a name should encode.

---

## 10. PEP 8 (Python naming)

**Overriding principle:** names visible to users as public parts of the API should reflect *usage*, not implementation.

| Thing | Convention | Example |
|---|---|---|
| Variables, functions, methods | `lower_case_with_underscores` | `clean_names()`, `order_total` |
| Classes | `CapWords` | `DataLoader` |
| Constants | `UPPER_CASE_WITH_UNDERSCORES` | `MAX_RETRIES` |
| Modules | short, lowercase, underscores allowed | `data_utils.py` |
| Packages | short, lowercase, underscores discouraged | `pandas` |
| Exceptions | CapWords, plus `Error` suffix if it's an error | `ValidationError` |
| Type variables | short CapWords | `T`, `AnyStr` |

### Underscore forms

- `_name`: weak "internal use" signal (not imported by `from module import *`).
- `name_`: trailing underscore to avoid a keyword clash (`class_`, `type_`).
- `__name`: name mangling in classes, to avoid subclass clashes.
- `__name__`: reserved for Python's own "magic" names. Never invent your own.

### Specific cautions

- Never use `l` (lowercase L), `O`, or `I` as single-character names.
- Acronyms in CapWords stay fully capitalized: `HTTPServerError`.
- `self` for instance methods and `cls` for class methods.
- On a keyword clash, prefer a trailing underscore over a misspelling (`class_`, not `klass`).
- Public data attributes shouldn't need getters and setters. Use a property if you need logic later.
- Anything undocumented or prefixed with `_` is internal.

### What it says about prefixes

PEP 8 mentions the idea of a short unique prefix to group related names (as the X11 library does) but notes it isn't much used in Python, because attribute and method names are already prefixed by the object and function names by the module. That is close to a direct statement against Hungarian-style prefixes.

### Related PEPs

- **PEP 484 (type hints):** strong argument for dropping types from names. `def get_name(user: User) -> str` carries the type in an enforced place.
- **PEP 20 (Zen of Python):** "Readability counts", "Explicit is better than implicit", "Namespaces are one honking great idea".

### Where data work bends or breaks PEP 8

- `X`, `X_train`, `X_test`: capital `X` follows math and scikit-learn convention, but linters flag it. Many teams allow it.
- `df`: near-universal and uninformative, so it fails "reflect usage".
- pandas **column names** are arbitrary strings, so PEP 8 doesn't govern them. That's where `Order Date` and `AmountUSD` get in, and the gap SQL style guides fill.
- PEP 8 itself says consistency within a project outranks consistency with the guide ("a foolish consistency is the hobgoblin of little minds").

---

## 11. SQL and database standards

### What is actually standardized

- The **SQL standard (ISO/IEC 9075)** defines the language but says little about naming style. Naming is almost entirely convention.
- Where dialects *differ* on identifiers:
  - **Case folding:** the standard folds unquoted identifiers to upper case, Postgres folds to lower, Snowflake folds to upper. Quoting makes a name case-sensitive forever.
  - **Length limits** differ by engine (see section 12). Long generated names can be silently truncated.
  - **Reserved words** differ. Columns named `user`, `order`, `date`, `year`, or `group` may need quoting.
- **Rule that follows:** lowercase `snake_case`, no spaces, never quote, and your names port across engines.

### ISO/IEC 11179: naming data elements

- A real standard for metadata registries. Pattern: *object class + property + representation term*, for example `customer` + `birth` + `date` giving `customer_birth_date`.
- You rarely implement it formally, but it's the source of the "entity, attribute, then a suffix saying what kind of value" habit (`_id`, `_date`, `_amount`, `_flag`, `_code`).
- Nuance to lean on: the representation term describes the **meaning** of the value (a date, an amount, a code), not a machine datatype.

### The practical standards are style guides

- Simon Holywell's *SQL Style Guide*
- GitLab data team's SQL style guide
- dbt style guide
- Joe Celko's *SQL Programming Style* (the book-length version)

**dbt conventions worth showing**

- Model layers: `stg_<source>__<entity>`, `int_`, `fct_`, `dim_`
- Primary keys named `<object>_id`
- Timestamps `<event>_at`, dates `<event>_date`, booleans `is_` / `has_`
- Plural table names; consistent column ordering (keys, then dates, then everything else)

**Formatting standards:** lowercase keywords, one column per line, explicit `JOIN` types, no `SELECT *` in production models, explicit aliases with `AS`.

### Database object naming (the unglamorous part)

- Constraints and indexes: `pk_orders`, `fk_orders_customer_id`, `uq_customers_email`, `ix_orders_created_at`. Without explicit names the database generates ones like `SYS_C0012345`, which are useless in error messages.
- **Smells to call out**
  - `tblCustomer`, `vwSales`: systems-Hungarian with the object type in the name.
  - SQL Server's `sp_` prefix on stored procedures, which makes the engine look in `master` first (a performance and ambiguity trap).
  - `_old`, `_backup`, `_v2`, `_new` tables: lifecycle in the name instead of version control or a dated schema.
- **Renames:** a warehouse column rename is a breaking change for every downstream dashboard and model. Use aliases or deprecation windows.

### Enforce it, don't just publish it

`sqlfluff` for SQL formatting and linting, dbt tests, `ruff` for Python, and pre-commit hooks. A standard that isn't enforced is a suggestion.

---

## 12. Limits on variable and identifier names

The limits explain much of the legacy: cryptic names like `cust_nm` mostly come from limits that no longer exist.

### Databases and SQL

| System | Max identifier length | Notes |
|---|---|---|
| SQL standard | Varies by version | SQL-92 had a very short limit (18 characters), raised later to 128 |
| PostgreSQL | 63 bytes | Longer names are **silently truncated** with only a notice |
| MySQL | 64 characters | Aliases can be longer |
| SQL Server / Azure SQL / Fabric Warehouse | 128 characters | |
| Oracle | 128 bytes (12.2 onward) | **30 bytes** before that, hence abbreviated older schemas |
| Snowflake | 255 characters | |
| BigQuery | Column names about 300 characters, table names much longer | Character rules loosened over time (flexible column names) |
| SQLite | No practical limit | |

### Languages and tools

| Tool | Limit | Notes |
|---|---|---|
| Python | No practical limit | PEP 8's 79-character line length is the real constraint |
| R | Very large (about 10,000 bytes) | Syntactic rules matter more than length |
| SAS | 32 characters | Was 8 in very old versions |
| Stata | 32 characters | |
| MATLAB | 63 characters | `namelengthmax` reports it |
| Excel | **31 characters for sheet names**, 255 for defined names | Classic trap when exporting |
| FORTRAN 77 / early C | 6 to 31 significant characters | Origin of terse naming habits |

### Rules that bite harder than length

- **Allowed characters:** most engines need a letter or underscore first, then letters, digits, underscores. Spaces, hyphens, and symbols need quoting or are rejected.
- **Case sensitivity:** unquoted names are folded, quoted ones are preserved.
- **Reserved words** vary by engine.
- **Unicode:** accepted by Python and most modern databases, but PEP 8 requires ASCII in the standard library, and non-ASCII names cause trouble in tooling, CSV exports, and BI tools.
- **Uniqueness after truncation:** in Postgres, two long names sharing their first 63 bytes collide.

### Where this causes real problems

- **Generated names:** constraint, index, and temp-table names are built from your names plus a suffix. A long dbt model name plus `__dbt_tmp` can exceed Postgres's limit, and the error may not point at the cause.
- **Cross-system pipelines:** names must survive the strictest system in the chain.
- **Spreadsheets and CSVs:** headers like `Order Date (USD)` break when they land in SQL or Parquet.

### The practical limit is human

Aim for under about 30 characters. Anything longer suggests the concept needs a definition, or the name is carrying context its table or schema should provide.

**For the talk:** pose "why are legacy schemas full of `cust_nm`?" and answer with Oracle's old 30-byte limit and SAS's 8-character limit. Three best slide facts: Postgres silent truncation, Excel's 31-character sheet names, Oracle's old 30-byte limit.

---

## 13. Power BI and Fabric

Microsoft publishes no single official naming standard. Guidance comes from Microsoft Learn, the Azure Cloud Adoption Framework, and community authors (SQLBI is the most cited).

### Core idea: two audiences, two styles

| Layer | Reader | Typical style | Example |
|---|---|---|---|
| Lakehouse / warehouse / SQL | Engineers | `snake_case`, no spaces, lowercase | `fct_sales`, `order_total_usd` |
| Semantic model (tables, columns, measures) | Business users in the field list | Title Case with spaces, plain language | `Sales[Order Total]`, `[Total Revenue]` |

The rename from `order_total_usd` to `Order Total` is a deliberate step at the boundary.

### Semantic model and DAX conventions

- **Spaces and Title Case are normal** here, unlike SQL. This forces quoting in DAX (`'Sales Order'[Order Date]`).
- **Singular dimension names, business nouns:** `Customer`, `Product`, `Date`. `Dim` / `Fact` prefixes are common in the warehouse but usually dropped in the model.
- **Hide plumbing** (keys, foreign keys, technical columns) instead of renaming it awkwardly.
- **The SQLBI convention** most worth showing:
  - Columns are always qualified: `Sales[Amount]`
  - Measures are never qualified: `[Total Sales]`

  The syntax tells you which you're looking at. It's the same idea as Apps Hungarian, with no type prefix needed.
- **A dedicated measures table** (often `_Measures`, with a leading underscore so it sorts to the top), organized into display folders.
- **Measure names are verb-free business terms:** `Total Revenue`, `Revenue YoY %`. Put units in the name only when ambiguous (`Revenue (USD)`).
- **Descriptions and synonyms matter more now.** Copilot and Q&A rely on model metadata.

### Power Query: self-documenting code in practice

- Every **applied step** has a name. The defaults (`Changed Type1`, `Filtered Rows2`) are Power Query's `df2`. Rename them to things like `Remove test accounts`.
- Staging queries are often marked with a prefix or underscore and set to not load.
- Parameters and functions get descriptive, verb-based names.

### Fabric items and workspaces

Fabric puts many item types in one flat workspace list. The community convention is a **type prefix** (`lh_`, `wh_`, `nb_`, `pl_`, `df_`), often combined with the medallion layer (`lh_bronze`, `lh_silver`, `lh_gold`).

This is a useful case for the types-in-names debate. A type prefix is usually wrong inside code, but here it earns its place:

- Item lists, search results, Git folders, and CLI or API output show names without icons.
- A lakehouse and a semantic model can legitimately share a business name.
- The Azure Cloud Adoption Framework recommends resource-type abbreviations for exactly this reason. It's deliberate policy.

Other items:

- **Workspaces** often carry environment and domain: `Sales - Dev`, `Sales - Prod`.
- **Deployment pipelines** pair items across stages, so renaming an item in one environment can cause confusion. Check current pairing behavior.
- **Lakehouse tables:** avoid spaces and special characters. Spark and Delta are case-sensitive in places the SQL analytics endpoint isn't, so lowercase `snake_case` is the safe default. Verify current collation and column-mapping behavior, since it has changed across releases.

### Why this is now a code problem

With the PBIP and TMDL formats and Fabric Git integration, semantic models and reports are text files in Git. Names show up in diffs, reviews, and merge conflicts. **Tabular Editor's Best Practice Analyzer** can enforce naming rules on a model.

**Slide idea: "Same company, three styles."** A column's journey: `order_total_usd` in the warehouse, `Order Total` in the model, `Revenue` on the visual.

---

## 14. Why a style guide matters

### How it makes reading faster

1. **Pattern recognition replaces parsing.** When every file looks the same, readers focus on logic. Experienced readers navigate by shape: where the joins, filters, and keys are.
2. **Predictability.** If timestamps are always `<event>_at`, you can *guess* a column name and be right.
3. **Less cognitive load.** Each inconsistency (`customer_id` here, `cust_key` there) makes the reader stop and ask whether they're the same thing.
4. **Cleaner diffs and reviews.** With automated formatting, pull requests show only real changes.
5. **Faster onboarding.** One set of rules reads across the whole repo.
6. **Bugs become visible.** A mismatched unit suffix or an unqualified column stands out.
7. **Ends bike-shedding.** The debate happens once, in the guide.

### Before and after for a slide

Inconsistent casing, aliases, and layout:

```sql
SELECT a.CustID, b.order_dt, SUM(b.AMT) tot
from customers a join Orders b on a.custid=b.CUSTID
where b.order_dt>'2025-01-01' group by a.CustID, b.order_dt
```

One guide (lowercase keywords, snake_case, meaningful aliases, one clause per line):

```sql
select
    customers.customer_id,
    orders.order_date,
    sum(orders.order_amount) as total_order_amount
from customers
inner join orders
    on customers.customer_id = orders.customer_id
where orders.order_date > '2025-01-01'
group by
    customers.customer_id,
    orders.order_date
```

Ask the room: "Which one can you find the join condition in faster?" For a rigorous claim you could cite program-comprehension and eye-tracking research on identifier style and layout, but look up specific studies before putting numbers on a slide.

### What a good style guide contains

- **Naming:** casing, word order, prefixes and suffixes, units, boolean and timestamp conventions, allowed or banned abbreviations.
- **Formatting:** indentation, line length, keyword case, trailing commas.
- **Structure:** CTE naming, model layering, file and folder layout.
- **Documentation rules:** what needs a comment (the *why*) and what needs a description.
- **Rationale for each rule**, and a short list of known exceptions.
- **Examples.** People copy examples more than they read rules.

### Making it stick

- **Automate it** with formatters and linters (`sqlfluff`, `black`, `ruff`).
- **Run it in pre-commit hooks and CI**, so non-conforming code never reaches review.
- **Keep it short.** One page with a linter beats thirty pages without one.
- **Version it** and treat changes like code changes.
- **Adopt before you invent.** Start from PEP 8, the dbt style guide, or the GitLab SQL guide, and change only what your team truly needs.

### Honest pitfalls

- **Consistency matters more than the specific choice.** Everyone using the same convention beats which convention it is.
- **Guides can become dogma.** A rule that makes code worse in a specific case should bend.
- **Retrofitting is costly.** Reformatting an old codebase creates giant diffs and ruins `git blame`. Adopt early, or do one clean reformat commit and ignore it in blame.
- **Style is not substance.** A perfectly formatted query can still be wrong. The guide is a floor, not a replacement for review.

---

## 15. Naming and style guides in the age of AI

### Why it matters more

1. **AI makes writing cheap, so reading is the bottleneck.** If an assistant produces 200 lines of SQL in seconds, the scarce resource is a human reviewer.
2. **Names are the context the model reads.** A model sees your schema, names, comments, and descriptions, not your business. `rev_adj_2` tells it nothing, and `net_revenue_usd` with a description lets it write correct joins and metrics. This applies to text-to-SQL, Copilot in Power BI, and coding assistants.
3. **Style guides become instructions to the AI.** Put the guide in files like `CLAUDE.md`, `AGENTS.md`, Copilot instructions, or Cursor rules so generated code follows house conventions the first time. A written, specific, example-driven guide is now a prompt as well as a document.
4. **Consistency helps models imitate you.** Models mirror surrounding patterns. A codebase with one clear convention gets consistent output.
5. **AI output can look right and be wrong.** Generated code is usually well-formatted and confident. Conventions ("measures are unqualified", "money columns end in `_usd`") help a reviewer spot errors fast.
6. **Enforcement is automatic.** Linters and formatters check AI output like human output, with no extra reviewer effort.
7. **Definitions still matter.** Descriptions, synonyms, and metric definitions give the model the *why* that a name can't carry.

### Nuance to include

- **The counter-argument:** models can infer meaning from messy code (`cust_nm` is probably a customer name), so you could argue naming matters less. The honest answer is that they often cope, but "often right" isn't good enough for a number in a board report, and inference errors are silent.
- **Names are no substitute for definitions.** Is `amount` gross or net, local currency or USD? A confidently misread column gives wrong answers that look authoritative.
- **AI can help fix naming.** Models are good at proposing renames, drafting descriptions, and applying a guide across a codebase. Humans still decide which names and definitions are right.
- **Don't over-claim.** No statistics about how much better AI performs with good naming unless cited. If you want numbers, look for published text-to-SQL evaluations with and without schema descriptions.

### Possible closing segment (about 4 minutes)

Three audiences for every name: **the human reading it today, the human maintaining it later, and the model generating from it.**

- A before/after: ask a model the same business question against a schema with cryptic names versus clear names plus descriptions. Run this for real before the talk so the result is honest and not staged.
- Add one item to the Monday action list: put your style guide where your AI tools can read it.

---

## 16. Takeaways (draft)

1. Names are the cheapest documentation you'll write, and they're read far more than they're written.
2. Let names and structure say *what*, and comments say *why*.
3. Encode **meaning** (unit, grain, state, representation), not storage types.
4. Pick a standard, write it down, and automate it. Lowercase `snake_case` with no quoting is the most portable default.
5. Rename early. Every dependent makes a rename costlier.
6. Put your style guide where your AI tools can read it.

### Monday-morning action list

1. Pick a base guide (PEP 8, the dbt style guide, or the GitLab SQL guide).
2. Add a linter and a pre-commit hook.
3. Write a one-page list of your team's naming decisions, with examples.
4. Point your AI tools at it.

---

## 17. Open questions and things to cut

**Still open**

- SQL/warehouse-weighted or Python/notebook-weighted? (This changes which examples lead.)
- Teach one rule of thumb, or present trade-offs and let people decide?
- What to cut. The material above exceeds 45 minutes.
- Whether to include other languages' guidance (R's tidyverse style guide, Google style guides, Go).
- Whether to add a section on running the conversation that produces a team's style guide.

**Candidates to trim if needed**

- Casing details (least interesting; keep to 2 minutes)
- The identifier-limits tables (keep only the three best facts)
- The Power BI / Fabric section (compress to the "same company, three styles" slide)
- Some of the data-model detail (units, grain, lifecycle)

**Possible deliverables, once you're ready (none started)**

- Slide deck (`.pptx`)
- Speaker notes with timing and anecdotes
- A runnable demo in this repo (bad vs good schema and query; the live-sort demo; the AI before/after test)
