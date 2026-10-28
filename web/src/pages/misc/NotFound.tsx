import { isRouteErrorResponse, useRouteError } from "react-router";
import { useT } from "../../i18n/i18n";
import { LinkButton } from "../../ui/Button";
import { Page, PageHead } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";

export function NotFound() {
  const t = useT();
  return (
    <Page>
      <PageHead title={t("Страница не найдена")} />
      <div>
        <LinkButton to="/">{t("На Стенд")}</LinkButton>
      </div>
    </Page>
  );
}

export function RouteError() {
  const t = useT();
  const error = useRouteError();
  if (isRouteErrorResponse(error) && error.status === 404) return <NotFound />;
  return (
    <Page>
      <PageHead title={t("Что-то пошло не так")} />
      <ErrorBox error={error instanceof Error ? error.message : error} />
      <div>
        <LinkButton to="/">{t("На Стенд")}</LinkButton>
      </div>
    </Page>
  );
}
