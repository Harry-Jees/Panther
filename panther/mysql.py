from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any

MYSQL_KEYRING_SERVICE = "panther-cli"
_MYSQL_ADMIN_STATEMENTS = {
    "GRANT",
    "REVOKE",
    "CREATE USER",
    "ALTER USER",
    "DROP USER",
    "RENAME USER",
    "CREATE ROLE",
    "ALTER ROLE",
    "DROP ROLE",
    "CREATE DATABASE",
    "ALTER DATABASE",
    "DROP DATABASE",
    "CREATE SCHEMA",
    "ALTER SCHEMA",
    "DROP SCHEMA",
    "SET",
    "USE",
    "START",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "RELEASE",
    "FLUSH",
    "RESET",
    "SHUTDOWN",
    "KILL",
    "INSTALL",
    "UNINSTALL",
    "LOAD",
    "LOCK",
    "UNLOCK",
}


def mysql_config_path(project_root: Path) -> Path:
    return project_root.resolve() / ".panther" / "mysql.json"


def mysql_keyring_username(project_root: Path) -> str:
    identity = str(project_root.resolve()).casefold().encode("utf-8")
    return f"mysql:{hashlib.sha256(identity).hexdigest()[:32]}"


def _validate_settings(settings: dict[str, Any]) -> None:
    if not isinstance(settings.get("host"), str) or not settings["host"].strip():
        raise ValueError("MySQL host must be a non-empty string")
    if (
        not isinstance(settings.get("port"), int)
        or isinstance(settings.get("port"), bool)
        or not 1 <= settings["port"] <= 65535
    ):
        raise ValueError("MySQL port must be an integer between 1 and 65535")
    for key in ("database", "username"):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"MySQL {key} must be a non-empty string")


def _mysql_connector() -> Any:
    try:
        return importlib.import_module("mysql.connector")
    except ImportError as exc:
        raise RuntimeError(
            "MySQL support requires the optional connector. Install Panther with "
            "`python -m pip install 'panther-cli[mysql]'`."
        ) from exc


def mysql_configuration(project_root: Path) -> dict[str, Any]:
    path = mysql_config_path(project_root)
    if not path.is_file():
        raise RuntimeError("MySQL is not configured. Run `panther mysql setup` first.")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read MySQL configuration {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"MySQL configuration must contain a JSON object: {path}")
    try:
        _validate_settings(config)
    except ValueError as exc:
        raise RuntimeError(f"MySQL configuration is incomplete or invalid: {exc}") from exc
    return config


def connect_mysql(
    project_root: Path,
    *,
    password: str | None = None,
    settings: dict[str, Any] | None = None,
) -> Any:
    settings = settings or mysql_configuration(project_root)
    _validate_settings(settings)
    if password is None:
        try:
            keyring = importlib.import_module("keyring")
            password = keyring.get_password(
                MYSQL_KEYRING_SERVICE,
                mysql_keyring_username(project_root),
            )
        except ImportError as exc:
            raise RuntimeError(
                "MySQL credentials require Panther's keyring dependency."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Could not read MySQL credentials from the OS keyring: {exc}") from exc
        if password is None:
            raise RuntimeError("MySQL credentials are missing. Run `panther mysql setup` again.")
    connector = _mysql_connector()
    try:
        return connector.connect(
            host=settings["host"],
            port=settings["port"],
            database=settings["database"],
            user=settings["username"],
            password=password,
            connection_timeout=5,
            read_timeout=30,
            write_timeout=30,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not connect to the configured MySQL database: {exc}") from exc


def _single_statement(query: Any) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty SQL string")
    if len(query) > 20_000:
        raise ValueError("query exceeds the 20,000-character limit")
    quote_char: str | None = None
    index = 0
    semicolon_seen = False
    while index < len(query):
        character = query[index]
        following = query[index + 1] if index + 1 < len(query) else ""
        if quote_char:
            if character == "\\" and quote_char != "`":
                index += 2
                continue
            if character == quote_char:
                if following == quote_char:
                    index += 2
                    continue
                quote_char = None
        elif character in {"'", '"', "`"}:
            quote_char = character
        elif character == "#" or (character == "-" and following == "-"):
            newline = query.find("\n", index)
            if newline < 0:
                break
            index = newline
        elif character == "/" and following == "*":
            end_comment = query.find("*/", index + 2)
            if end_comment < 0:
                raise ValueError("query contains an unterminated SQL comment")
            index = end_comment + 1
        elif character == ";":
            if semicolon_seen or query[index + 1:].strip():
                raise ValueError("Only one SQL statement may be executed at a time")
            semicolon_seen = True
        index += 1
    if quote_char:
        raise ValueError("query contains an unterminated SQL string or identifier")
    return query.strip().rstrip(";").strip()


def _statement_keyword(query: str) -> str:
    without_comments = re.sub(r"/\*.*?\*/|--[^\r\n]*|#[^\r\n]*", " ", query, flags=re.DOTALL)
    return without_comments.lstrip().split(None, 1)[0].upper() if without_comments.strip() else ""


def _bounded_select(query: str) -> str:
    return f"SELECT * FROM ({query}) AS `_panther_bounded_result` LIMIT 101"


def _is_admin_statement(query: str) -> bool:
    normalized = re.sub(r"/\*.*?\*/|--[^\r\n]*|#[^\r\n]*", " ", query, flags=re.DOTALL)
    normalized = " ".join(normalized.upper().split())
    if any(
        normalized == statement or normalized.startswith(statement + " ")
        for statement in _MYSQL_ADMIN_STATEMENTS
    ):
        return True
    if re.match(
        r"^(?:CREATE|ALTER|DROP)(?: OR REPLACE)?\s+(?:USER|ROLE|DATABASE|SCHEMA)\b",
        normalized,
    ):
        return True
    return any(
        normalized.startswith(statement)
        for statement in (
            "CREATE USER",
            "ALTER USER",
            "DROP USER",
            "RENAME USER",
            "CREATE ROLE",
            "ALTER ROLE",
            "DROP ROLE",
            "CREATE DATABASE",
            "ALTER DATABASE",
            "DROP DATABASE",
            "LOAD DATA",
            "LOCK TABLES",
            "UNLOCK TABLES",
        )
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytesHex": value[:512].hex(), "truncated": len(value) > 512}
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + "…[truncated]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _parameters(args: dict[str, Any]) -> tuple[Any, ...]:
    parameters = args.get("parameters", [])
    if not isinstance(parameters, list) or len(parameters) > 100:
        raise ValueError("parameters must be a list containing no more than 100 values")
    if any(
        not isinstance(value, (str, int, float, bool, type(None)))
        for value in parameters
    ):
        raise ValueError("SQL parameters must be JSON scalar values")
    return tuple(parameters)


def mysql_read(args: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Execute a single read-only query and return a bounded result set."""
    connection = None
    cursor = None
    try:
        query = _single_statement(args.get("query"))
        keyword = _statement_keyword(query)
        if keyword not in {"SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}:
            raise ValueError("mysql_read accepts only SELECT, SHOW, DESCRIBE, DESC, or EXPLAIN")
        if re.search(
            r"\b(?:INTO\s+(?:OUTFILE|DUMPFILE)|FOR\s+UPDATE|LOCK\s+IN\s+SHARE\s+MODE|"
            r"LOAD_FILE\s*\(|GET_LOCK\s*\(|RELEASE_LOCK\s*\(|SLEEP\s*\(|BENCHMARK\s*\()",
            query,
            flags=re.IGNORECASE,
        ):
            raise ValueError("Locking and file-writing clauses are not allowed in read-only queries")
        connection = connect_mysql(project_root)
        connection.start_transaction(readonly=True)
        cursor = connection.cursor()
        bounded_query = (
            _bounded_select(query)
            if keyword == "SELECT"
            else query
        )
        cursor.execute(bounded_query, _parameters(args))
        columns = [str(item[0]) for item in (cursor.description or ())]
        rows = cursor.fetchmany(101) if columns else []
        limited = len(rows) > 100
        result = {
            "columns": columns[:100],
            "rows": [
                [_json_value(value) for value in row[:100]]
                for row in rows[:100]
            ],
            "truncated": limited or len(columns) > 100,
        }
        rendered = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        while len(rendered) > 30_000 and result["rows"]:
            result["rows"].pop()
            result["truncated"] = True
            rendered = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        return {"success": True, "data": rendered}
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": f"MySQL read query failed: {exc}"}
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            try:
                connection.rollback()
            finally:
                connection.close()


def mysql_execute(args: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Execute one user-approved data or schema change and commit it."""
    connection = None
    cursor = None
    try:
        query = _single_statement(args.get("query"))
        keyword = _statement_keyword(query)
        if not keyword:
            raise ValueError("query must contain one SQL statement")
        if _is_admin_statement(query):
            raise ValueError(f"MySQL administrative statement '{keyword}' is not permitted")
        connection = connect_mysql(project_root)
        cursor = connection.cursor()
        cursor.execute(query, _parameters(args))
        affected = cursor.rowcount
        last_id = cursor.lastrowid
        connection.commit()
        return {
            "success": True,
            "data": json.dumps(
                {"affectedRows": affected, "lastInsertId": last_id},
                separators=(",", ":"),
            ),
        }
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        if connection is not None:
            connection.rollback()
        return {"success": False, "error": f"MySQL statement failed: {exc}"}
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def test_mysql_connection(
    project_root: Path,
    password: str | None = None,
    settings: dict[str, Any] | None = None,
) -> None:
    connection = connect_mysql(project_root, password=password, settings=settings)
    try:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT 1")
            row = cursor.fetchone()
            if not row or row[0] != 1:
                raise RuntimeError("MySQL connection test returned an unexpected result.")
        finally:
            cursor.close()
    finally:
        connection.close()
