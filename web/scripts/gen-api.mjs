// Типы клиента из OpenAPI сервера: поднимает create_app() через uv (без сети и без сервера),
// пишет схему во временный файл и генерирует src/api/schema.d.ts. Файл лежит в репозитории:
// сборка UI не требует Python.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(web, "..");
const tmp = mkdtempSync(join(tmpdir(), "ragglet-openapi-"));

const python = `
import json, sys, tempfile
from ragglet.api.app import create_app
from ragglet.api.settings import ServerSettings
tmp = tempfile.mkdtemp()
settings = ServerSettings(_env_file=None, artifacts_dir=tmp + "/runs", builds_dir=tmp + "/builds")
json.dump(create_app(settings).openapi(), sys.stdout, ensure_ascii=False, indent=1)
`;

try {
  const spec = execFileSync("uv", ["run", "--quiet", "python", "-c", python], {
    cwd: repo,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  const specPath = join(tmp, "openapi.json");
  writeFileSync(specPath, spec);
  execFileSync(join(web, "node_modules/.bin/openapi-typescript"), [specPath, "-o", "src/api/schema.d.ts"], {
    cwd: web,
    stdio: "inherit",
  });
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
