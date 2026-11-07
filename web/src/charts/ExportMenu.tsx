import type { RefObject } from "react";
import { useT } from "../i18n/i18n";
import { Button } from "../ui/Button";
import { Menu } from "../ui/Menu";
import type { ChartExport } from "./Chart";

/** «⤓» в шапке карточки графика: SVG или PNG, всегда в светлой теме (решение 3 ревью). */
export function ExportMenu({ target }: { target: RefObject<ChartExport | null> }) {
  const t = useT();
  return (
    <Menu
      trigger={
        <Button small aria-label={t("Скачать график")} title={t("Скачать график: SVG или PNG, светлая тема")}>
          ⤓
        </Button>
      }
      items={[
        { label: "SVG", onSelect: () => void target.current?.download("svg") },
        { label: "PNG", onSelect: () => void target.current?.download("png") },
      ]}
    />
  );
}

/** Кнопка «CSV» в шапке карточки таблицы. */
export function CsvButton({ onClick }: { onClick: () => void }) {
  const t = useT();
  return (
    <Button small onClick={onClick} title={t("Скачать таблицу в CSV")}>
      CSV
    </Button>
  );
}
