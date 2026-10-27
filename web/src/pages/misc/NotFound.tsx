import { isRouteErrorResponse, useRouteError } from "react-router";
import { LinkButton } from "../../ui/Button";
import { Page, PageHead } from "../../ui/Layout";
import { ErrorBox } from "../../ui/ProblemView";

export function NotFound() {
  return (
    <Page>
      <PageHead title="Страница не найдена" />
      <div>
        <LinkButton to="/">На Стенд</LinkButton>
      </div>
    </Page>
  );
}

export function RouteError() {
  const error = useRouteError();
  if (isRouteErrorResponse(error) && error.status === 404) return <NotFound />;
  return (
    <Page>
      <PageHead title="Что-то пошло не так" />
      <ErrorBox error={error instanceof Error ? error.message : error} />
      <div>
        <LinkButton to="/">На Стенд</LinkButton>
      </div>
    </Page>
  );
}

export function Soon({ title }: { title: string }) {
  return (
    <Page>
      <PageHead title={title} />
      <p className="muted">Раздел появится на следующем этапе.</p>
    </Page>
  );
}
